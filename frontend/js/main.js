// Sets up each section of the page from the package's catalog and connects
// them: loading a dataset shows it everywhere, defining the selection decides
// whether it can run, and a running selection locks the rest of the page

import {setupCharts, showData, showFeatures} from "./charts.js";
import {setDatasetLocked, setupDataset} from "./dataset.js";
import {
    readSelection, setSelectionLocked, setSubsetSize, setupSelection
} from "./selection.js";
import {
    resetOptimization, setReady, setupOptimization
} from "./optimization.js";
import {
    resetTransforms, setTransformsLocked, setupTransforms
} from "./transforms.js";
// the tooltips of the page's own controls, by element id or control name
import tooltips from "/json/tooltips.json" with {type: "json"};

// the sections shown once a dataset is loaded
const sections = [
    "datasetInformation", "datasetTransforms", "defineSelection",
    "optimization"
].map(id => document.getElementById(id));

/**
 * Show a newly loaded dataset, clearing the last selection and the record
 * of transforms of the data it replaces, and reveal the sections that use it
 * @param {Object} reply The dataset's feature names, size and data by feature
 */
function showDataset(reply) {
    resetTransforms();
    showTransformed(reply);
}

/**
 * Show the dataset's current data, after loading or transforming it,
 * clearing the last selection
 * @param {Object} reply The dataset's feature names, size and data by feature
 */
function showTransformed(reply) {
    const [rows, columns] = reply.size;
    resetOptimization();
    setSubsetSize(rows, columns);
    for (const section of sections) section.hidden = false;
    showFeatures(reply.features);
    showData(reply.data);
}

/**
 * Lock or unlock the dataset and the definition of the selection, while a
 * selection runs
 * @param {boolean} running Whether a selection is running
 */
function lockProblem(running) {
    setDatasetLocked(running);
    setTransformsLocked(running);
    setSelectionLocked(running);
}

// the tooltips of the controls written in the page, by their element ids
for (const [id, tooltip] of Object.entries(tooltips)) {
    const element = document.getElementById(id);
    if (element && tooltip) element.title = tooltip;
}

// the objectives, algorithms, generators and other entries of the package
const catalog = await (await fetch("/catalog")).json();

setupCharts();
setupDataset(catalog, showDataset);
setupTransforms(catalog, showTransformed);
setupSelection(catalog, count => setReady(count > 0));
setupOptimization(catalog, readSelection, lockProblem);
