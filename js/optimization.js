// The Optimization section: the algorithm, running and stopping a selection,
// and following its progress from the server's event stream

import {
    fitted, infoField, lockControls, options, readSettings, settingInputs
} from "./fields.js";
import {clearLossChart, showLoss, showSubset, startLoss} from "./charts.js";

const optimizationControls = document.getElementById("optimizationControls");
const algorithm = document.getElementById("algorithm");
const algorithmSettings = document.getElementById("algorithmSettings");
const selectSubset = document.getElementById("selectSubset");
const stopSelection = document.getElementById("stopSelection");
const selectionInfo = document.getElementById("selectionInfo");

let algorithms = {}; // each algorithm's settings, from the catalog
let improvements = []; // the [evaluation, time, loss] of each new best subset
let readSelection = () => ({}); // reads the selection as defined
let lockOthers = () => {}; // locks the other sections while a run goes on

/**
 * Set up choosing an algorithm from the catalog and running selections
 * @param {Object} catalog The package's catalog from the server
 * @param {Function} read Reads the selection as defined, for a request
 * @param {Function} lock Locks or unlocks the other sections' controls
 */
export function setupOptimization(catalog, read, lock) {
    algorithms = catalog.algorithms;
    readSelection = read;
    lockOthers = lock;

    algorithm.append(...options(Object.keys(algorithms)));
    fitted(algorithm);
    showAlgorithmSettings();
    algorithm.addEventListener("change", showAlgorithmSettings);

    selectSubset.addEventListener("click", select);
    stopSelection.addEventListener("click", () => {
        fetch("/stop", {method: "POST"});
    });

    // follow the progress of selections the server publishes
    const events = new EventSource("/events");
    events.addEventListener("message", follow);
}

/**
 * Show Select Subset only when a subset can be selected
 * @param {boolean} ready Whether there is at least one objective
 */
export function setReady(ready) {
    selectSubset.hidden = !ready;
}

/**
 * Clear the last selection's information and loss chart, for a new dataset
 */
export function resetOptimization() {
    selectionInfo.textContent = "";
    clearLossChart();
}

/**
 * Show the settings of the chosen algorithm
 */
function showAlgorithmSettings() {
    const settings = settingInputs(algorithms[algorithm.value]);
    algorithmSettings.replaceChildren(...settings);
}

/**
 * Ask the server to select a subset as defined, with the chosen algorithm,
 * starting a fresh loss chart for its progress
 */
async function select() {
    const request = {
        ...readSelection(),
        algorithm: algorithm.value,
        settings: readSettings(algorithmSettings),
    };

    selectionInfo.textContent = "Selecting...";
    selectionInfo.classList.remove("error");
    setRunning(true);
    improvements = [];
    startLoss();

    const response = await fetch("/select", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(request),
    });
    if (!response.ok) {
        const reply = await response.json();
        setRunning(false);
        selectionInfo.classList.add("error");
        selectionInfo.textContent =
            `Could not select a subset: ${reply.error}`;
    }
}

/**
 * Show a selection's progress from a message of the server's event stream,
 * with its best subset so far on the scatterplot and its loss on the chart
 * @param {MessageEvent} event The message, as JSON
 */
function follow(event) {
    const message = JSON.parse(event.data);
    const ended = message.status !== "improved";
    if (ended) setRunning(false);
    if (message.status === "failed") {
        selectionInfo.classList.add("error");
        selectionInfo.textContent = `Selection failed: ${message.error}`;
        return;
    }

    if (message.status === "improved") {
        improvements.push([message.evaluation, message.time, message.loss]);
    }
    showSubset(new Set(message.subset));
    showLoss(improvements, message.evaluation, message.time);
    // the share of the declared total done so far stands in for Running,
    // when the algorithm declared one
    const {total, evaluation} = message;
    const progress = total ? `${Math.round(100 * evaluation / total)}%` : null;
    const status = {
        improved: progress ?? "Running",
        stopped: progress ? `Stopped at ${progress}` : "Stopped",
        finished: "Done",
    };
    selectionInfo.replaceChildren(
        infoField("Loss", message.loss.toFixed(3)),
        infoField("Evaluations", message.evaluation),
        infoField("Time (s)", message.time.toFixed(3)),
        infoField("Status", status[message.status]),
    );
}

/**
 * Lock every control of the dataset and the problem while a selection runs,
 * putting Stop in Select Subset's place, and unlock them when it ends
 * @param {boolean} running Whether a selection is running
 */
function setRunning(running) {
    lockControls(optimizationControls, running);
    selectSubset.hidden = running;
    stopSelection.hidden = !running;
    lockOthers(running); // unlocking shows Select Subset again if it can run
}
