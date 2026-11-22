# --- Imports ------------------------------------------------------------------

# standard library
import inspect
import io
import json
import queue
import re
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator, Literal, get_args, get_origin
from urllib.parse import parse_qs, urlparse

# third party
import numpy as np

# local files
from . import algorithm, generate, metric, objective, scalarization, transform
from .dataset import Dataset
from .loss import Loss
from .problem import Problem
from .results import Results
from .solver import Solver

# --- Package Options ----------------------------------------------------------

def describeArguments(constructor) -> dict[str, str]:
    """
    Read the description of each argument of a constructor from the Args
    section of its docstring, with wrapped lines joined

    Args:
        constructor: The class or function whose docstring to read
    """
    function = (
        constructor.__init__ if inspect.isclass(constructor) else constructor
    )
    text = inspect.getdoc(function) or ""
    descriptions = {}
    name = None
    inArguments = False
    for line in text.splitlines():
        if line.strip() == "Args:":
            inArguments = True
        elif inArguments:
            if line and not line.startswith(" "): # the next section
                break
            argument = re.match(r"    (\w+): (.+)", line)
            if argument:
                name = argument.group(1)
                descriptions[name] = argument.group(2).strip()
            elif name and line.startswith("        ") and line.strip():
                descriptions[name] += " " + line.strip() # a wrapped line
    return descriptions


def buildCatalog() -> dict[str, dict[str, dict]]:
    """
    Build the catalog of all objectives, algorithms, dataset generators,
    metrics and scalarizations and their available settings
    """
    # the public functions or classes of each module, by catalog section
    kinds = {
        "objectives": (objective, inspect.isclass),
        "algorithms": (algorithm, inspect.isclass),
        "generators": (generate, inspect.isfunction),
        "metrics": (metric, inspect.isfunction),
        "scalarizations": (scalarization, inspect.isclass),
        "transforms": (transform, inspect.isfunction),
    }
    sections = {}
    for section, (module, kind) in kinds.items():
        sections[section] = {
            name: member for name, member in inspect.getmembers(module, kind)
            if member.__module__ == module.__name__ # defined, not imported
            and not name.startswith("_")
        }

    # the metric functions in definition order, and if they give a scalar
    sections["metrics"] = dict(sorted(
        sections["metrics"].items(),
        key=lambda m: m[1].__code__.co_firstlineno
    ))
    metrics = sections["metrics"]
    scalar = [
        name for name, function in metrics.items()
        if inspect.signature(function).return_annotation is float
    ]

    # describe each constructor's settings for the page from its signature
    catalog = {}
    for section, constructors in sections.items():
        catalog[section] = {}
        for entry, constructor in constructors.items():
            parameters = inspect.signature(constructor).parameters.items()
            descriptions = describeArguments(constructor)
            if section == "metrics": # metrics are chosen, not set up
                parameters = []
            settings = {}
            for name, parameter in parameters:
                if section == "scalarizations" and name == "weights":
                    continue
                if name == "representation":
                    continue

                # a transform's data comes from the dataset, and the page
                # applies every transform to every feature
                if section == "transforms" and name in ("data", "indices"):
                    continue

                # the default value, or None if the setting is required
                required = parameter.default is inspect.Parameter.empty
                default = None if required else parameter.default
                norm = name == "p" # a norm order, with infinity sent as "inf"
                choices = [1, 2, "inf"] if norm else None

                # a metric is chosen by name, starting from the first defined
                if name == "metric":
                    choices = list(metrics)
                    default = choices[0]

                annotation = parameter.annotation
                types = get_args(annotation) or (annotation,)
                if get_origin(annotation) is Literal: # named choices
                    choices = list(types)
                integer = int in types and float not in types
                pair = get_origin(annotation) is tuple
                settings[name] = {
                    "default": default,
                    "choices": choices,
                    "integer": integer,
                    "pair": pair,
                    "description": descriptions.get(name)
                }
                if name == "metric":
                    settings[name]["scalar"] = scalar
            catalog[section][entry] = {
                "constructor": constructor,
                "settings": settings
            }
    return catalog

# --- Server and Handler Classes -----------------------------------------------

class Server(ThreadingHTTPServer):
    """
    Serves the package as an interactive web application
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

    def transform(self, request: dict) -> None:
        """
        Transform a representation of the loaded dataset as described by a
        page request, into another representation or over the data itself

        Args:
            request: The transform, chosen by name, with its settings, the
                name it reads ("from") and the name it stores to ("into")

        Raises:
            KeyError: If the transform, the name read, or a field is not found
            TypeError: If a setting is missing or not accepted
            ValueError: If no dataset is loaded, the name stored to is empty,
                or the package rejects a value
        """
        if self.dataset is None:
            raise ValueError("No dataset loaded")
        if not request["into"]:
            raise ValueError("The name to store into is empty")

        entry = self.catalog["transforms"][request["transform"]]
        source = self.dataset[request["from"]]
        self.dataset[request["into"]] = entry["constructor"](
            source, **request["settings"]
        )

    def select(self, request: dict) -> dict:
        """
        Start selecting a subset of the loaded dataset as described by a page
        request, in a worker thread that publishes its progress to listeners

        Args:
            request: The objectives with their weights, the scalarization
                that combines them, the subset size, and the algorithm with
                its settings, chosen by name

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
            if "metric" in values: # chosen by name
                metrics = self.catalog["metrics"]
                values["metric"] = metrics[values["metric"]]["constructor"]
            entry = self.catalog["objectives"][o["name"]]
            objectives.append(entry["constructor"](**values))
        weights = [o["weight"] for o in chosen]
        entry = self.catalog["scalarizations"][request["scalarization"]]
        loss = Loss(objectives, entry["constructor"](weights))
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
                        "total": results.total,
                        "time": results.time,
                        "loss": results.loss,
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
            "time": results.time
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
        elif self.path.startswith("/representation?"): # one array, by name
            query = parse_qs(urlparse(self.path).query)
            name = query.get("name", [""])[0]
            try:
                array = self.server.dataset[name]
            except (KeyError, TypeError) as e: # TypeError: no dataset loaded
                self.sendJSON(400, {"error": str(e)})
                return
            # the dataset's feature names only fit an array of the same width
            same = array.shape[1] == self.server.dataset.size[1]
            features = (self.server.dataset.features if same
                        else [str(i) for i in range(array.shape[1])])
            self.sendJSON(200, {
                "features": features,
                "data": array.T.tolist()
            })
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
        if self.path == "/dataset": # Send the dataset loaded from file
            file = self.rfile.read(int(self.headers["Content-Length"])).decode()
            try:
                self.server.dataset = Dataset(io.StringIO(file))
            except ValueError as e:
                self.send_error(400, explain=str(e))
                return
            self.sendData()
        elif self.path == "/generate": # Send a synthetically generated  dataset
            body = self.rfile.read(int(self.headers["Content-Length"]))
            try:
                self.server.generate(json.loads(body))
            except (KeyError, TypeError, ValueError) as e:
                self.sendJSON(400, {"error": str(e)})
                return
            self.sendData()
        elif self.path == "/transform": # Send the transformed representation
            body = self.rfile.read(int(self.headers["Content-Length"]))
            try:
                self.server.transform(json.loads(body))
            except (KeyError, TypeError, ValueError) as e:
                self.sendJSON(400, {"error": str(e)})
                return
            self.sendData()
        elif self.path == "/remove": # Send the removal information
            body = self.rfile.read(int(self.headers["Content-Length"]))
            try:
                del self.server.dataset[json.loads(body)["name"]]
            except (KeyError, TypeError, ValueError) as e:
                self.sendJSON(400, {"error": str(e)})
                return
            self.sendJSON(200, {"removed": True})
        elif self.path == "/select": # Send the selection information
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
            "data": dataset["data"].T.tolist()
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