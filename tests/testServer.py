# --- Imports ------------------------------------------------------------------

# standard library
import json
import queue
import threading
import time
import unittest
from functools import partial
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

# local files
from visselect import Loss, Problem, Solver, metric, objective, scalarization
from visselect.algorithm import Greedy, Random
from visselect.server import Handler, Server, buildCatalog


class QuietHandler(Handler):
    """
    The server's handler with its per-request terminal logging silenced
    """

    def log_message(self, *args) -> None:
        """Log nothing to keep the test output clean"""


# --- Test Classes -------------------------------------------------------------

class TestBuildCatalog(unittest.TestCase):
    """
    Test building the catalog of what the package offers
    """

    def setUp(self):
        """Build the catalog of the package"""
        self.catalog = buildCatalog()

    def testObjectiveMetric(self):
        """Test that a metric is a setting chosen from the metrics by name"""
        settings = self.catalog["objectives"]["Preserve"]["settings"]
        self.assertEqual(settings["metric"]["choices"],
                         ["mean", "variance", "range", "max", "min", "hull"])
        self.assertEqual(settings["metric"]["default"], "mean")
        self.assertEqual(settings["metric"]["scalar"], ["hull"])
        function = self.catalog["metrics"]["variance"]["constructor"]
        built = self.catalog["objectives"]["Preserve"]["constructor"](
            metric=function
        )
        self.assertIsInstance(built, objective.Preserve)
        self.assertIs(built._metric, metric.variance)

    def testObjectiveSettings(self):
        """Test that a norm has its choices"""
        settings = self.catalog["objectives"]["Preserve"]["settings"]
        norm = {
            "default": 1, "choices": [1, 2, "inf"],
            "integer": True, "pair": False,
            "description": "The order of the norm of the difference "
                "between metrics: 1, 2, ∞"
        }
        self.assertEqual(settings["p"], norm)

    def testAlgorithms(self):
        """Test that algorithms are found with their settings and defaults"""
        algorithms = self.catalog["algorithms"]
        self.assertIs(algorithms["Greedy"]["constructor"], Greedy)
        self.assertIs(algorithms["Random"]["constructor"], Random)
        whole = {"choices": None, "integer": True, "pair": False}
        self.assertEqual(algorithms["Random"]["settings"], {
            "samples": {"default": 1, "description":
                "The number of random samples to take and evaluate"} | whole,
            "seed": {"default": None, "description":
                "The random seed for reproducibility"} | whole,
        })

    def testScalarizations(self):
        """Test that scalarizations are found, their weights left to the page"""
        entry = self.catalog["scalarizations"]["WeightedSum"]
        self.assertIs(entry["constructor"], scalarization.WeightedSum)
        self.assertEqual(entry["settings"], {})

    def testDescriptions(self):
        """Test that a setting carries its docstring description"""
        settings = self.catalog["generators"]["blobs"]["settings"]
        self.assertEqual(settings["balance"]["description"],
                         "How balanced the clusters' shares of items are, "
                         "0 to 1")

    def testGenerators(self):
        """Test that generators are found with whole and decimal pairs"""
        settings = self.catalog["generators"]["blobs"]["settings"]
        self.assertTrue(settings["size"]["pair"])
        self.assertTrue(settings["size"]["integer"])
        self.assertTrue(settings["centerInterval"]["pair"])
        self.assertFalse(settings["centerInterval"]["integer"])
        self.assertFalse(settings["balance"]["pair"])

class TestServer(unittest.TestCase):
    """
    Test the web application server
    """

    def setUp(self):
        """Create a server on an ephemeral port with one event listener"""
        self.server = Server("127.0.0.1", 0, Handler)
        self.addCleanup(self.server.server_close)
        self.listener = queue.Queue()
        self.server.listeners.append(self.listener)
        self.request = {
            "objectives": [{"name": "Preserve", "weight": 1.0,
                            "settings": {"metric": "mean", "p": "inf"}},
                           {"name": "Distinctness", "weight": 1.0,
                            "settings": {}}],
            "scalarization": "WeightedSum",
            "size": 5,
            "algorithm": "Random",
            "settings": {"samples": 10, "seed": 0},
        }

    def messages(self):
        """The messages published to the listener so far, drained in order"""
        drained = []
        while not self.listener.empty():
            drained.append(self.listener.get())
        return drained

    def testInit(self):
        """Test that the server binds its socket and starts with clean state"""
        server = Server("127.0.0.1", 0, Handler)
        self.addCleanup(server.server_close)

        host, port = server.server_address
        self.assertEqual(host, "127.0.0.1")
        self.assertGreater(port, 0) # port 0 asks the OS for an ephemeral port

        self.assertIsNone(server.dataset)
        self.assertIsNone(server.worker)
        self.assertEqual(server.listeners, [])
        self.assertFalse(server.stopping.is_set())
        self.assertEqual(server.catalog.keys(), buildCatalog().keys())

        # both guard locks start released
        for lock in (server.starting, server.listening):
            self.assertTrue(lock.acquire(blocking=False))
            lock.release()

    def testSelect(self):
        """Test that a selection runs and publishes improvements and its end"""
        self.server.generate({"generator": "normal",
                              "settings": {"size": (20, 2), "seed": 0}})
        reply = self.server.select(self.request)
        self.assertEqual(reply, {"started": True})
        self.assertIsInstance(self.server.method, Random) # held for stopping
        self.server.worker.join(timeout=10)
        self.assertFalse(self.server.worker.is_alive())

        messages = self.messages()
        first, last = messages[0], messages[-1]
        self.assertEqual(first["status"], "improved")
        self.assertEqual(first["evaluation"], 1)
        self.assertEqual(first["total"], 10)
        self.assertGreaterEqual(first["time"], 0.0)
        self.assertEqual(len(first["subset"]), 5)
        self.assertEqual(last["status"], "finished")
        self.assertEqual(last["evaluation"], 10)

        # improvements are published only when the loss gets lower
        losses = [message["loss"] for message in messages]
        self.assertEqual(losses, sorted(losses, reverse=True))

    def testSelectNoDataset(self):
        """Test that selecting with no dataset loaded is rejected"""
        with self.assertRaisesRegex(ValueError, "No dataset"):
            self.server.select(self.request)

    def testTransform(self):
        """Test that a transform stores a new representation of the dataset"""
        self.server.generate({"generator": "normal",
                              "settings": {"size": (20, 2), "seed": 0}})
        self.server.transform({"transform": "scale", "from": "data",
                               "into": "scaled", "settings": {}})
        scaled = self.server.dataset["scaled"]
        self.assertEqual(scaled.shape, (20, 2))
        self.assertEqual(scaled.min(), 0.0)
        self.assertEqual(scaled.max(), 1.0)

    def testTransformReplaces(self):
        """Test that transforming into a stored name fells its branch"""
        self.server.generate({"generator": "normal",
                              "settings": {"size": (20, 2), "seed": 0}})
        self.server.transform({"transform": "scale", "from": "data",
                               "into": "scaled", "settings": {}})
        self.server.transform({"transform": "discretize", "from": "scaled",
                               "into": "binned", "settings": {"bins": 2}})
        self.assertEqual(self.server.dataset.node("binned").parent.name,
                         "scaled")
        self.server.transform({"transform": "scale", "from": "data",
                               "into": "scaled", "settings": {}})
        self.assertNotIn("binned", self.server.dataset)

    def testTransformData(self):
        """Test that transforming into data replaces it and fells the tree"""
        self.server.generate({"generator": "normal",
                              "settings": {"size": (20, 2), "seed": 0}})
        self.server.transform({"transform": "scale", "from": "data",
                               "into": "scaled", "settings": {}})
        self.server.transform({"transform": "scale", "from": "scaled",
                               "into": "data", "settings": {}})
        self.assertEqual(self.server.dataset["data"].min(), 0.0)
        self.assertNotIn("scaled", self.server.dataset)

    def testTransformRejects(self):
        """Test that transforming without a dataset or a name is rejected"""
        request = {"transform": "scale", "from": "data",
                   "into": "", "settings": {}}
        with self.assertRaisesRegex(ValueError, "No dataset"):
            self.server.transform(request)
        self.server.generate({"generator": "normal",
                              "settings": {"size": (20, 2), "seed": 0}})
        with self.assertRaisesRegex(ValueError, "empty"):
            self.server.transform(request)

    def testSelectWhileRunning(self):
        """Test that a second selection is rejected while one is running"""
        self.server.generate({"generator": "normal",
                              "settings": {"size": (20, 2), "seed": 0}})
        release = threading.Event()
        self.server.worker = threading.Thread(target=release.wait)
        self.server.worker.start()
        self.addCleanup(self.server.worker.join)
        self.addCleanup(release.set) # cleanups run last-in-first-out
        with self.assertRaisesRegex(ValueError, "already running"):
            self.server.select(self.request)

    def testStopped(self):
        """Test that a stopped selection publishes its stopped status"""
        self.server.generate({"generator": "normal",
                              "settings": {"size": (20, 2), "seed": 0}})
        problem = Problem(self.server.dataset,
                          Loss([objective.Preserve(metric.mean)]), 5)
        self.server.stop() # stopping before the run stops its first evaluation
        self.server.run(Solver(Random(samples=10, seed=0)), problem)

        messages = self.messages()
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["status"], "improved")
        self.assertEqual(messages[-1]["status"], "stopped")
        self.assertEqual(messages[-1]["evaluation"], 1)

    def testFailed(self):
        """Test that a failing selection publishes its error"""
        def broken(problem, emit):
            raise ValueError("broken algorithm")
        self.server.run(Solver(broken), None)
        self.assertEqual(self.messages(),
                         [{"status": "failed", "error": "broken algorithm"}])

    def testStopInterrupts(self):
        """Test that stop also interrupts an algorithm that supports it"""
        stops = []
        class Stoppable:
            def stop(self):
                stops.append(True)
        self.server.method = Stoppable()
        self.server.stop()
        self.assertTrue(self.server.stopping.is_set())
        self.assertEqual(stops, [True])

    def testEndedWithoutImprovements(self):
        """Test that a selection with no evaluations still publishes its end"""
        def idle(problem, emit):
            pass
        self.server.run(Solver(idle), None)
        [message] = self.messages()
        self.assertEqual(message["status"], "finished")
        self.assertEqual(message["evaluation"], 0)

class TestHandler(unittest.TestCase):
    """
    Test the HTTP request handling of the web application server
    """

    def setUp(self):
        """Serve on an ephemeral port in a background thread"""
        root = Path(__file__).resolve().parents[1]
        self.server = Server("127.0.0.1", 0,
                             partial(QuietHandler, directory=str(root)))
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown) # cleanups run last-in-first-out
        threading.Thread(
            target=partial(self.server.serve_forever, poll_interval=0.01),
            daemon=True
        ).start()
        self.address = f"http://127.0.0.1:{self.server.server_address[1]}"

    def get(self, path):
        """Send a GET request, returning the status code and the JSON reply"""
        try:
            with urlopen(self.address + path, timeout=5) as reply:
                return reply.status, json.loads(reply.read())
        except HTTPError as error:
            return error.code, json.loads(error.read())

    def post(self, path, body):
        """Send a POST request, returning the status code and the JSON reply"""
        if not isinstance(body, str):
            body = json.dumps(body)
        request = Request(self.address + path, data=body.encode())
        try:
            with urlopen(request, timeout=5) as reply:
                return reply.status, json.loads(reply.read())
        except HTTPError as error:
            body = error.read() # error pages from send_error are html, not JSON
            return error.code, json.loads(body) if body.startswith(b"{") else None

    def generate(self):
        """Load a generated dataset over HTTP, returning the reply"""
        return self.post("/generate", {"generator": "normal",
                                       "settings": {"size": [20, 2], "seed": 0}})

    def testIndex(self):
        """Test that the root path serves the application page"""
        with urlopen(self.address + "/", timeout=5) as reply:
            self.assertEqual(reply.status, 200)
            self.assertIn(b"<html", reply.read(200).lower())

    def testCatalog(self):
        """Test that the catalog is served as JSON with settings only"""
        status, catalog = self.get("/catalog")
        self.assertEqual(status, 200)
        self.assertEqual(catalog.keys(), buildCatalog().keys())
        self.assertTrue(catalog["algorithms"]["Random"]["samples"]["integer"])

    def testGenerate(self):
        """Test that generating serves the new dataset back, by feature"""
        status, reply = self.generate()
        self.assertEqual(status, 200)
        self.assertEqual(reply["size"], [20, 2])
        self.assertEqual(len(reply["data"]), 2) # transposed: a row per feature
        self.assertEqual(len(reply["data"][0]), 20)

        status, reply = self.post("/generate", {"generator": "nope",
                                                "settings": {}})
        self.assertEqual(status, 400)
        self.assertIn("error", reply)

    def testDataset(self):
        """Test that a posted csv file loads and rejects unreadable data"""
        status, reply = self.post("/dataset", "a,b\n1.0,2.0\n3.0,4.0\n")
        self.assertEqual(status, 200)
        self.assertEqual(reply["features"], ["a", "b"])
        self.assertEqual(reply["size"], [2, 2])

        status, reply = self.post("/dataset", "a,b\nx,y\n")
        self.assertEqual(status, 400)

    def testTransform(self):
        """Test that transforming serves the dataset and rejects bad names"""
        self.generate()
        request = {"transform": "scale", "from": "data",
                   "into": "scaled", "settings": {}}
        status, reply = self.post("/transform", request)
        self.assertEqual(status, 200)
        self.assertEqual(reply["size"], [20, 2])

        status, reply = self.post("/transform", request | {"transform": "nope"})
        self.assertEqual(status, 400)
        self.assertIn("error", reply)

    def testTree(self):
        """Test that the tree route serves each node's name and lineage"""
        self.assertEqual(self.get("/tree"), (200, []))
        self.generate()
        self.post("/transform", {"transform": "scale", "from": "data",
                                 "into": "scaled", "settings": {}})
        status, reply = self.get("/tree")
        self.assertEqual(status, 200)
        self.assertEqual(reply, [{
            "name": "scaled", "parent": "data", "transform": "scale",
            "settings": {"interval": [0.0, 1.0], "indices": None}
        }])

    def testRepresentation(self):
        """Test that a representation is served by name with its features"""
        self.generate()
        status, reply = self.get("/representation?name=data")
        self.assertEqual(status, 200)
        self.assertEqual(len(reply["features"]), 2)
        self.assertEqual(len(reply["data"]), 2)

        status, reply = self.get("/representation?name=missing")
        self.assertEqual(status, 400)
        self.assertIn("error", reply)

    def testRemove(self):
        """Test that removing a representation serves back the confirmation"""
        self.generate()
        self.post("/transform", {"transform": "scale", "from": "data",
                                 "into": "scaled", "settings": {}})
        status, reply = self.post("/remove", {"name": "scaled"})
        self.assertEqual(status, 200)
        self.assertEqual(reply, {"removed": True})

        status, reply = self.post("/remove", {"name": "missing"})
        self.assertEqual(status, 400)

    def testSelect(self):
        """Test that selecting starts the worker and rejects bad requests"""
        self.generate()
        status, reply = self.post("/select", {
            "objectives": [{"name": "Preserve", "weight": 1.0,
                            "settings": {"metric": "mean"}}],
            "scalarization": "WeightedSum",
            "size": 5,
            "algorithm": "Random",
            "settings": {"samples": 10, "seed": 0},
        })
        self.assertEqual(status, 200)
        self.assertEqual(reply, {"started": True})
        self.server.worker.join(timeout=10)
        self.assertFalse(self.server.worker.is_alive())

        status, reply = self.post("/select", {})
        self.assertEqual(status, 400)
        self.assertIn("error", reply)

    def testStop(self):
        """Test that stopping is confirmed and signals the selection"""
        status, reply = self.post("/stop", "")
        self.assertEqual(status, 200)
        self.assertEqual(reply, {"stopping": True})
        self.assertTrue(self.server.stopping.is_set())

    def testNotFound(self):
        """Test that an unknown POST path is a 404"""
        status, reply = self.post("/nope", "")
        self.assertEqual(status, 404)

    def testEvents(self):
        """Test that published messages stream to a listening page"""
        stream = urlopen(self.address + "/events", timeout=5)
        self.addCleanup(stream.close)
        self.server.publish({"ping": 1})
        self.assertEqual(stream.readline(), b'data: {"ping": 1}\n')
        self.assertEqual(stream.readline(), b"\n")

        stream.close() # the next writes to the page fail, dropping its queue
        for _ in range(100):
            self.server.publish({"bye": 1})
            if not self.server.listeners:
                break
            time.sleep(0.01)
        self.assertEqual(self.server.listeners, [])


if __name__ == "__main__":
    unittest.main()
