// The Define Selection section: the subset size, the scalarization, and the
// objectives written as the terms of the loss

import {
    fitted, label, lockControls, options, readSettings, tip
} from "./fields.js";
import {formulaParts, isNegative} from "./formula.js";

const selectionControls = document.getElementById("selectionControls");
const objectives = document.getElementById("objectives");
const addObjective = document.getElementById("addObjective");
const subsetSize = document.getElementById("subsetSize");
const subsetFeatures = document.getElementById("subsetFeatures");
const scalarization = document.getElementById("scalarization");

let objectiveSettings = {}; // each objective's settings, from the catalog
let onChange = () => {}; // called with the number of objectives on a change

/**
 * Set up defining the selection from the objectives and scalarizations in
 * the catalog
 * @param {Object} catalog The package's catalog from the server
 * @param {Function} changed Called with the number of objectives whenever
 *     they are added or removed
 */
export function setupSelection(catalog, changed) {
    objectiveSettings = catalog.objectives;
    onChange = changed;

    addObjective.append(...options(Object.keys(objectiveSettings)));
    addObjective.addEventListener("change", addTerm);
    scalarization.append(...options(Object.keys(catalog.scalarizations)));
    fitted(scalarization);
    fitted(subsetSize);
}

/**
 * Set the subset size for a newly loaded dataset, at most one fewer than its
 * items, and show its number of features, which a subset keeps
 * @param {number} rows The dataset's number of items
 * @param {number} columns The dataset's number of features
 */
export function setSubsetSize(rows, columns) {
    subsetSize.max = rows - 1;
    subsetSize.value = Math.min(10, rows - 1);
    subsetSize.dispatchEvent(new Event("input")); // refit it
    subsetFeatures.textContent = columns;
}

/**
 * Read the selection as defined, for a request to select a subset
 * @returns {Object} The objectives with their weights and settings, the
 *     scalarization and the subset size
 */
export function readSelection() {
    return {
        objectives: [...objectives.children].map(row => ({
            name: row.dataset.objective,
            weight: Number(row.querySelector(".weight input").value),
            settings: readSettings(row),
        })),
        scalarization: scalarization.value,
        size: Number(subsetSize.value),
    };
}

/**
 * Lock the section's controls while a selection runs, hiding Add Objective,
 * and unlock them
 * @param {boolean} running Whether a selection is running
 */
export function setSelectionLocked(running) {
    lockControls(selectionControls, running); // also locks the typeset values
    addObjective.hidden = running;
    if (!running) updateObjectives();
}

/**
 * Add the chosen objective as a term of the loss L(S), its weight λ in front
 */
function addTerm() {
    const name = addObjective.value;
    const row = document.createElement("div");
    row.className = "term";
    row.dataset.objective = name;
    if (isNegative(name)) row.dataset.negative = "";

    // L(S) = before the first term and + before the rest, typeset by
    // updateObjectives as terms are added and removed
    const lead = document.createElement("span");
    lead.className = "lead math";

    const formula = document.createElement("span");
    formula.className = "formula fieldValue";
    formula.append(...formulaParts(name, objectiveSettings[name]));

    // which objective the term is, by name, after the term like a note
    const objectiveName = document.createElement("span");
    objectiveName.className = "tokenLabel fieldName";
    objectiveName.textContent = "Objective";
    const objective = document.createElement("span");
    objective.className = "objective";
    objective.append(objectiveName, label(name));

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "remove";
    remove.title = tip("remove");
    remove.textContent = "×";
    remove.addEventListener("click", () => {
        row.remove();
        updateObjectives();
    });

    row.append(lead, weightField(), formula, objective, remove);
    objectives.append(row);
    addObjective.selectedIndex = 0;
    updateObjectives();
}

/**
 * Create a term's weight λ, typeset as a number, which clicking swaps for an
 * input until it is entered, since MathML cannot hold an input
 * @returns {HTMLSpanElement} The weight's field
 */
function weightField() {
    const weight = document.createElement("input");
    weight.type = "number";
    weight.min = 0;
    weight.step = "any";
    weight.value = 1;
    weight.title = tip("weight");
    weight.className = "editable";
    const typed = fitted(weight);
    typed.hidden = true;
    const typeset = document.createElement("span");
    typeset.className = "math";
    typeset.title = tip("weight");
    const show = () => {
        typeset.innerHTML = "<math><mn class=\"editable\"></mn></math>";
        typeset.querySelector("mn").textContent = weight.value || "0";
    };
    show();
    weight.addEventListener("input", show);
    typeset.addEventListener("click", () => {
        if (weight.disabled) return; // locked while a selection runs
        typeset.hidden = true;
        typed.hidden = false;
        weight.focus();
        weight.select();
    });
    weight.addEventListener("blur", () => {
        typed.hidden = true;
        typeset.hidden = false;
    });
    weight.addEventListener("keydown", event => {
        if (event.key === "Enter") weight.blur();
    });
    const weightName = document.createElement("span");
    weightName.className = "fieldName tokenLabel";
    weightName.textContent = "λ";
    const weightValue = document.createElement("span");
    weightValue.className = "weight fieldValue";
    weightValue.append(weightName, typeset, typed);
    return weightValue;
}

/**
 * Write L(S) = before the first term and + or − before the rest, and show
 * weights only when blending two or more objectives
 */
function updateObjectives() {
    const rows = objectives.children;
    for (const [i, row] of [...rows].entries()) {
        // a negative term's minus takes the place of the +, or follows the =
        const negative = "negative" in row.dataset;
        row.querySelector(".lead").innerHTML = i === 0
            ? "<math><mi>L</mi><mo>⁡</mo><mo stretchy=\"false\">(</mo>"
                + "<mi>S</mi><mo stretchy=\"false\">)</mo><mo>=</mo>"
                + (negative ? "<mo>−</mo>" : "") + "</math>"
            : `<math><mo>${negative ? "−" : "+"}</mo></math>`;

        const weightLabel = row.querySelector(".weight");
        weightLabel.hidden = rows.length < 2;
        if (weightLabel.hidden) {
            const weight = weightLabel.querySelector("input");
            weight.value = 1;
            weight.dispatchEvent(new Event("input")); // refit it
        }
    }
    onChange(rows.length);
}
