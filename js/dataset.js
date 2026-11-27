// The Load Dataset section: importing a file or generating a synthetic dataset

import {
    fitted, infoField, lockControls, options, readSettings, settingInputs, tip
} from "./fields.js";

const loadDataset = document.getElementById("loadDataset");
const datasetChoice = document.getElementById("datasetChoice");
const datasetInput = document.getElementById("datasetInput");
const randomGeneration = document.getElementById("randomGeneration");
const generatorButtons = document.getElementById("generatorButtons");
const generatorControls = document.getElementById("generatorControls");
const generator = document.getElementById("generator");
const generatorSettings = document.getElementById("generatorSettings");
const generateDataset = document.getElementById("generateDataset");
const fileButtons = document.getElementById("fileButtons");
const fileControls = document.getElementById("fileControls");
const fileInfo = document.getElementById("fileInfo");
const fileToGeneration = document.getElementById("fileToGeneration");
const datasetInfo = document.getElementById("datasetInfo");

let generators = {}; // each generator's settings, from the catalog
let onLoad = () => {}; // called with each newly loaded dataset

/**
 * Set up loading a dataset, from a file or a generator in the catalog
 * @param {Object} catalog The package's catalog from the server
 * @param {Function} loaded Called with each dataset loaded, as the server's
 *     reply of its feature names, size and data by feature
 */
export function setupDataset(catalog, loaded) {
    generators = catalog.generators;
    onLoad = loaded;

    generator.append(...options(Object.keys(generators)));
    generator.value = "blobs"; // start with clusters, the clearest structure
    fitted(generator);
    showGeneratorSettings();
    generator.addEventListener("change", showGeneratorSettings);

    datasetInput.addEventListener("change", loadFile);
    generateDataset.addEventListener("click", generate);

    // switch to the options for random generation, from the start or a
    // file, with the distribution chosen up front and still changeable
    for (const toGeneration of [randomGeneration, fileToGeneration]) {
        toGeneration.append(...options(Object.keys(generators)));
        toGeneration.addEventListener("change", () => {
            generator.value = toGeneration.value;
            generator.dispatchEvent(new Event("change")); // show settings
            toGeneration.selectedIndex = 0;
            showMode("generate");
        });
    }
}

/**
 * Lock the section's controls while a selection runs, hiding its buttons
 * while keeping their space so the page does not shift, and unlock them
 * @param {boolean} running Whether a selection is running
 */
export function setDatasetLocked(running) {
    lockControls(loadDataset, running); // also locks the file name
    for (const button of loadDataset.querySelectorAll(".button")) {
        button.style.visibility = running ? "hidden" : "";
    }
}

/**
 * Show the buttons in the section's heading and the controls below it for
 * one way of loading a dataset
 * @param {string} mode "start", "generate" or "file"
 */
function showMode(mode) {
    datasetChoice.hidden = mode !== "start";
    generatorButtons.hidden = mode !== "generate";
    generatorControls.hidden = mode !== "generate";
    fileButtons.hidden = mode !== "file";
    fileControls.hidden = mode !== "file";
}

/**
 * Show the settings of the chosen dataset generator
 */
function showGeneratorSettings() {
    const settings = settingInputs(generators[generator.value]);
    generatorSettings.replaceChildren(...settings);
}

/**
 * Load the chosen file as the dataset and show its name and size, where
 * clicking the name, which is a label of the file input, chooses another
 */
async function loadFile() {
    const datasetFile = datasetInput.files[0];
    if (!datasetFile) return;

    const response = await fetch("/dataset", {
        method: "POST",
        headers: {"Content-Type": "text/csv"},
        body: datasetFile,
    });
    datasetInput.value = ""; // so choosing the same file again loads it again
    if (!response.ok) {
        datasetInfo.textContent = "Could not load file";
        datasetInfo.classList.add("error");
        console.error(response.status, await response.text());
        return;
    }
    const reply = await response.json();

    const [rows, columns] = reply.size;
    const fileName = infoField("File", "");
    const choose = document.createElement("label");
    choose.className = "fileName editable";
    choose.htmlFor = datasetInput.id;
    choose.title = tip("fileName");
    choose.textContent = datasetFile.name;
    fileName.querySelector(".fieldValue").append(choose);
    const size = infoField("Dataset Size", `${rows} × ${columns}`);
    fileInfo.replaceChildren(fileName, size);
    showMode("file");
    loaded(reply);
}

/**
 * Generate a dataset with the chosen generator and settings and load it
 */
async function generate() {
    const request = {
        generator: generator.value,
        settings: readSettings(generatorSettings),
    };
    const response = await fetch("/generate", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(request),
    });
    const reply = await response.json();
    if (!response.ok) {
        datasetInfo.textContent = `Could not generate data: ${reply.error}`;
        datasetInfo.classList.add("error");
        return;
    }
    loaded(reply);
}

/**
 * Pass a newly loaded dataset on, clearing any message from a failed load
 * @param {Object} reply The dataset's feature names, size and data by feature
 */
function loaded(reply) {
    datasetInfo.textContent = "";
    datasetInfo.classList.remove("error");
    onLoad(reply);
}
