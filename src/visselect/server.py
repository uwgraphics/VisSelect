# --- Imports ------------------------------------------------------------------

# standard library
import inspect
import io
import json
import queue
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator, get_args, get_origin

# third party
import numpy as np

# local files
from . import algorithm, generate, metric, objective
from .dataset import Dataset
from .loss import Loss
from .problem import Problem
from .results import Results
from .scalarization import WeightedSum
from .solver import Solver

# --- Package Options ----------------------------------------------------------

def buildCatalog() -> dict[str, dict[str, dict]]:
    """
    Build the catalog of all objectives, algorithms and dataset generators and
    their available settings
    """
    # the public functions or classes defined in each module of the package
    searches = [
        (metric, inspect.isfunction),
        (objective, inspect.isclass),
        (algorithm, inspect.isclass),
        (generate, inspect.isfunction),
    ]

    public = {}
    for module, kind in searches:
        public[module] = {}
        for name, member in inspect.getmembers(module, kind):
            defined = member.__module__ == module.__name__ # not imported
            private = name.startswith("_")
            if defined and not private:
                public[module][name] = member

    # expand objectives that take a metric into one objective per metric
    objectives = {}
    for name, constructor in public[objective].items():
        parameters = inspect.signature(constructor).parameters
        if "metric" in parameters:
            for metricName, function in public[metric].items():
                bound = {"metric": function}
                # a scalar metric's difference is the same under every norm
                returns = inspect.signature(function).return_annotation
                if returns is float and "p" in parameters:
                    bound["p"] = 1
                objectives[f"{name} {metricName}"] = partial(
                    constructor, **bound
                )
        else:
            objectives[name] = constructor

    # describe each constructor's settings for the page from its signature
    sections = {
        "objectives": objectives,
        "algorithms": public[algorithm],
        "generators": public[generate],
    }
    catalog = {}
    for section, constructors in sections.items():
        catalog[section] = {}
        for entry, constructor in constructors.items():
            parameters = inspect.signature(constructor).parameters.items()
            fixed = getattr(constructor, "keywords", {}) # set by partial
            settings = {}
            for name, parameter in parameters:
                if name in fixed: # such as the metric of an expanded objective
                    continue
                
                # the default value, or None if the setting is required
                required = parameter.default is inspect.Parameter.empty
                default = None if required else parameter.default
                norm = name == "p" # a norm order, with infinity sent as "inf"
                choices = [1, 2, "inf"] if norm else None

                # whole or decimal numbers, from the annotation, and whether
                # the setting takes a pair such as an interval
                annotation = parameter.annotation
                types = get_args(annotation) or (annotation,)
                integer = int in types and float not in types
                pair = get_origin(annotation) is tuple
                settings[name] = {
                    "default": default,
                    "choices": choices,
                    "integer": integer,
                    "pair": pair
                }
            catalog[section][entry] = {
                "constructor": constructor,
                "settings": settings
            }
    return catalog

# --- Server and Handler Classes -----------------------------------------------

class Server(ThreadingHTTPServer):
    """
    Serves the package as an interactive webapp
    """
    def __init__(self, host, port, handler):
        super().__init__((host, port), handler)
        self.dataset = None
        self.catalog = buildCatalog()

        self.worker = None # the thread running the latest selection
        self.starting = threading.Lock() # so only one selection runs at a time
        self.stopping = threading.Event() # set to stop the running selection
        self.listeners = [] # a message queue for each page's event stream
        self.listening = threading.Lock() # guards adding and removing queues

    def generate(self, request: dict) -> None:
        """
        Load a randomly generated dataset as described by a page request

        Args:
            request: The generator, chosen by name, with its settings

        Raises:
            KeyError: If the generator or a field is not found
            TypeError: If a setting is missing or not accepted
            ValueError: If the package rejects a value
        """
        entry = self.catalog["generators"][request["generator"]]
        data = entry["constructor"](**request["settings"])
        self.dataset = Dataset(data)

    def select(self, request: dict) -> dict:
        """
        Start selecting a subset of the loaded dataset as described by a page
        request, in a worker thread that publishes its progress to listeners

        Args:
            request: The objectives with their weights, the subset size, and
                the algorithm with its settings, chosen by name

        Raises:
            KeyError: If an objective, algorithm or field is not found
            TypeError: If an algorithm setting is not accepted
            ValueError: If no dataset is loaded, a selection is already
                running, or the package rejects a value
        """
        if self.dataset is None:
            raise ValueError("No dataset loaded")

        chosen = request["objectives"]
        objectives = []
        for o in chosen:
            values = o["settings"].items()
            values = {k: np.inf if v == "inf" else v for k, v in values}
            entry = self.catalog["objectives"][o["name"]]
            objectives.append(entry["constructor"](**values))
        weights = [o["weight"] for o in chosen]
        loss = Loss(objectives, WeightedSum(weights))
        problem = Problem(self.dataset, loss, request["size"])

        entry = self.catalog["algorithms"][request["algorithm"]]
        method = entry["constructor"](**request["settings"])
        steps = Solver(method).steps(problem)

        with self.starting:
            if self.worker is not None and self.worker.is_alive():
                raise ValueError("A selection is already running")
            self.stopping.clear()
            self.worker = threading.Thread(
                target=self.run, args=(steps,), daemon=True
            )
            self.worker.start()

        return {"started": True}

    def run(self, steps: Iterator[Results]) -> None:
        """
        Run the steps of a selection in the worker thread until it finishes or
        is stopped, then publish its outcome, or its error if it fails
        """
        best = None # the frozen subset last published, replaced on improvement
        stopped = False
        try:
            for evaluation, results in enumerate(steps, start=1):
                if results.subset is not best:
                    best = results.subset
                    improved = {
                        "status": "improved",
                        "evaluation": evaluation,
                        "time": results._time,
                        "loss": results._loss,
                        "subset": np.flatnonzero(best.indicator).tolist()
                    }
                    self.publish(improved)
                if self.stopping.is_set():
                    steps.close()
                    stopped = True
                    break
        except Exception as e:
            self.publish({"status": "failed", "error": str(e)})
            return

        # the best loss and subset are those of the last improvement
        status = "stopped" if stopped else "finished"
        ended = {
            "status": status,
            "evaluation": evaluation,
            "time": results._time
        }
        self.publish(improved | ended)

    def stop(self) -> None:
        """
        Stop the running selection after its current evaluation
        """
        self.stopping.set()

    def publish(self, message: dict) -> None:
        """
        Send a message to the event stream of every connected page
        """
        with self.listening:
            for listener in self.listeners:
                listener.put(message)


class Handler(SimpleHTTPRequestHandler):
    """
    Handles the incoming and outgoing HTTP requests for a client on the server
    """
    def do_GET(self):
        """
        Handles GET HTTP requests to the server
        """
        if self.path == "/":
            self.path = "/html/index.html"
        if self.path == "/events":
            listener = queue.Queue()
            with self.server.listening:
                self.server.listeners.append(listener)
    
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
    
            try:
                while True:
                    message = listener.get()
                    event = f"data: {json.dumps(message)}\n\n"
                    self.wfile.write(event.encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError): # the page closed
                pass
            finally:
                with self.server.listening:
                    self.server.listeners.remove(listener)
        elif self.path == "/catalog": # the settings of each entry, as JSON
            catalog = {}
            for section, entries in self.server.catalog.items():
                catalog[section] = {
                    name: entry["settings"] for name, entry in entries.items()
                }
            self.sendJSON(200, catalog)
        else:
            super().do_GET()

    def do_POST(self):
        """
        Handles POST HTTP responses to a client using the server
        """
        if self.path == "/dataset":
            file = self.rfile.read(int(self.headers["Content-Length"])).decode()
            try:
                self.server.dataset = Dataset(io.StringIO(file))
            except ValueError as e:
                self.send_error(400, explain=str(e))
                return
            self.sendData()
        elif self.path == "/generate":
            body = self.rfile.read(int(self.headers["Content-Length"]))
            try:
                self.server.generate(json.loads(body))
            except (KeyError, TypeError, ValueError) as e:
                self.sendJSON(400, {"error": str(e)})
                return
            self.sendData()
        elif self.path == "/select":
            body = self.rfile.read(int(self.headers["Content-Length"]))
            try:
                reply = self.server.select(json.loads(body))
            except (KeyError, TypeError, ValueError) as e:
                self.sendJSON(400, {"error": str(e)})
                return
            self.sendJSON(200, reply)
        elif self.path == "/stop":
            self.server.stop()
            self.sendJSON(200, {"stopping": True})
        else:
            self.send_error(404)

    def sendData(self) -> None:
        """
        Sends the loaded dataset's features, size and data, by feature
        """
        dataset = self.server.dataset
        self.sendJSON(200, {
            "features": dataset.features,
            "size": dataset.size,
            "data": dataset.data.T.tolist()
        })

    def sendJSON(self, code: int, message) -> None:
        """
        Sends a response to the client with a message encoded as JSON
        """
        body = json.dumps(message).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

if __name__ == "__main__":
    hostName = "localhost"
    serverPort = 8080
    path = Path(__file__).resolve().parents[2] / "frontend"

    handler = partial(Handler, directory=path)

    with Server(hostName, serverPort, handler) as server:
        print("Server Started")

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass

        print("Server Stopped")