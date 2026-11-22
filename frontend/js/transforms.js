// The Apply Transforms section: a tree of representations of the dataset,
// drawn as a graph once a transform is applied; + on a node sets up a
// transform of that representation and × removes a leaf, and while a
// transform is set up its row takes the graph's place until it is applied

import {
    fitted, infoField, label, lockControls, options, readSettings,
    settingInputs, tip
} from "./fields.js";

const section = document.getElementById("datasetTransforms");
const addTransform = document.getElementById("transform");
const applyTransform = document.getElementById("applyTransform");

const radius = 16; // the size of a node's circle
const columnGap = 100; // the space from a node to its children
const rowGap = 64; // the space between sibling nodes

let transforms = {}; // each transform's settings, from the catalog
let onApply = () => {}; // called when a transform changes the data itself
let nodes = []; // each representation: its name, parent and provenance
let dataDetails = []; // the transforms applied over the data itself
let graph = null; // the div holding the drawn graph, once one is drawn
let note = null; // the graph's error message
let pending = null; // the row being set up, until it is applied or removed
let nextRowId = 0; // numbers the inputs so their labels can name them

/**
 * Set up applying the transforms in the catalog to the dataset
 * @param {Object} catalog The package's catalog from the server
 * @param {Function} applied Called when a transform replaces the data
 *     itself, with the server's reply of its features, size and data
 */
export function setupTransforms(catalog, applied) {
    transforms = catalog.transforms;
    onApply = applied;

    addTransform.hidden = false;
    addTransform.addEventListener("click", () => addRow("data"));
    applyTransform.addEventListener("click", apply);
}

/**
 * The names an objective can evaluate a subset on: the data itself and
 * every stored representation, in the order they were created
 * @returns {string[]} The names
 */
export function representationNames() {
    return ["data", ...nodes.map(node => node.name)];
}

/**
 * Remove the whole tree, drawn or being set up, when a new dataset
 * replaces the representations it described
 */
export function resetTransforms() {
    for (const row of section.querySelectorAll("p")) row.remove();
    nodes = [];
    dataDetails = [];
    if (graph) graph.remove();
    graph = null;
    pending = null;
    addTransform.hidden = false;
    applyTransform.hidden = true;
    announce();
}

/**
 * Tell the rest of the page that the stored representations changed, so
 * controls naming them can follow
 */
function announce() {
    document.dispatchEvent(new CustomEvent("representationsChanged"));
}

/**
 * Lock the section's controls while a selection runs, hiding its buttons
 * while keeping their space so the page does not shift, and unlock them;
 * the graph's + and × hide through the section's running style
 * @param {boolean} running Whether a selection is running
 */
export function setTransformsLocked(running) {
    lockControls(section, running);
    for (const button of section.querySelectorAll(".button")) {
        button.style.visibility = running ? "hidden" : "";
    }
}

/**
 * Add a row to set up a transform of one representation, taking the place
 * of the graph until it is applied from the heading or removed with its ×
 * @param {string} source The name of the representation the transform reads
 */
function addRow(source) {
    const row = document.createElement("p");
    const rowId = nextRowId++;

    const choice = document.createElement("select");
    choice.className = "editable";
    choice.id = `transform${rowId}`;
    choice.append(...options(Object.keys(transforms)));

    // the source the + was clicked on, kept in view while the graph is not
    const from = infoField("From", source);

    // the result's name starts as the transform's natural branch name, and
    // follows the transform choice until a name is typed over it
    const into = document.createElement("input");
    into.type = "text";
    into.className = "editable";
    into.id = `transformInto${rowId}`;
    let autoName = `${choice.value}d`;
    into.value = autoName;
    choice.addEventListener("change", () => {
        const name = `${choice.value}d`;
        if (into.value === autoName) {
            into.value = name;
            into.dispatchEvent(new Event("input")); // refit it
        }
        autoName = name;
    });

    // the setting fields sit directly in the row, so the last field of the
    // row is the only one without a trailing gap
    const showSettings = () => {
        for (const old of row.querySelectorAll("[data-transform-setting]")) {
            old.remove();
        }
        const fields = settingInputs(transforms[choice.value]);
        for (const field of fields) {
            field.setAttribute("data-transform-setting", "");
        }
        from.before(...fields);
    };
    choice.addEventListener("change", showSettings);

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "remove";
    remove.title = tip("removeTransform");
    remove.textContent = "×";
    remove.addEventListener("click", () => {
        row.remove();
        showGraph();
    });

    const message = document.createElement("span");
    message.className = "error";

    row.append(
        rowField("Transform", choice, "transformChoice"), from,
        rowField("Name", into, "transformInto"), remove, " ", message
    );
    showSettings();
    section.append(row);
    pending = {row, choice, source, into, message};
    if (graph) graph.hidden = true;
    addTransform.hidden = true;
    applyTransform.hidden = false;
}

/**
 * Create one of the row's own fields: its name above its control
 * @param {string} name The name shown above the control
 * @param {HTMLElement} control The editable control, with an id
 * @param {string} tooltip The name of the control's tooltip
 * @returns {HTMLSpanElement} The field
 */
function rowField(name, control, tooltip) {
    control.title = tip(tooltip);
    const fieldName = document.createElement("label");
    fieldName.className = "fieldName";
    fieldName.textContent = name;
    fieldName.htmlFor = control.id;
    const fieldValue = document.createElement("span");
    fieldValue.className = "fieldValue";
    fieldValue.append(fitted(control));
    const field = document.createElement("span");
    field.className = "field";
    field.append(fieldName, " ", fieldValue, " ");
    return field;
}

/**
 * Apply the pending transform and return to the graph with its result as a
 * node, or show why it was rejected; a transform stored over "data" fells
 * every representation, on the server and so in the graph
 */
async function apply() {
    const {row, choice, source, into, message} = pending;
    const name = into.value.trim();
    if (!name) {
        message.textContent = "Name is empty";
        return;
    }
    const values = readSettings(row);
    const response = await fetch("/transform", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            transform: choice.value, settings: values,
            from: source, into: name,
        }),
    });
    const reply = await response.json();
    if (!response.ok) {
        message.textContent = `Could not apply: ${reply.error}`;
        return;
    }

    const parts = Object.entries(values).map(
        ([setting, value]) => `${label(setting)} ${shown(value)}`);
    const detail =
        [label(choice.value), ...parts, `from ${source}`].join(", ");
    if (name === "data") {
        nodes = []; // replacing the data fells every representation
        dataDetails.push(detail);
    } else {
        const existing = nodes.find(node => node.name === name);
        if (existing) existing.details.push(detail);
        else nodes.push({name, parent: source, details: [detail]});
    }

    row.remove();
    showGraph();
    announce();
    if (name === "data") onApply(reply);
}

/**
 * Draw and show the graph of the representation tree, with the heading's
 * buttons put away, since the nodes' + take over adding transforms
 */
function showGraph() {
    pending = null;
    applyTransform.hidden = true;
    const drawn = nodes.length > 0 || dataDetails.length > 0;
    addTransform.hidden = drawn;
    if (!drawn) { // the last node went: back to the starting button
        if (graph) graph.hidden = true;
        return;
    }

    if (!graph) {
        graph = document.createElement("div");
        graph.id = "transformGraph";
        note = document.createElement("span");
        note.className = "error";
        section.append(graph);
    }
    graph.hidden = false;
    note.textContent = "";

    // lay the tree out with the data at the left, children rightward
    const children = {};
    for (const node of nodes) (children[node.parent] ??= []).push(node);
    const toTree = name => ({
        name, children: (children[name] ?? []).map(child => toTree(child.name))
    });
    const root = d3.hierarchy(toTree("data"));
    d3.tree().nodeSize([rowGap, columnGap])(root);
    const places = root.descendants();
    const low = Math.min(...places.map(place => place.x));
    const margin = {top: radius + 20, left: 2}; // room for the top names
    const x = place => margin.left + radius + place.y;
    const y = place => margin.top + place.x - low;
    const width = margin.left + root.height * columnGap + 2 * radius + 110;
    const height = margin.top + Math.max(...places.map(place => place.x))
        - low + radius + 24;

    const svg = d3.create("svg").attr("width", width).attr("height", height);
    for (const place of places.slice(1)) {
        svg.append("line")
            .attr("class", "edge")
            .attr("x1", x(place.parent) + radius)
            .attr("y1", y(place.parent))
            .attr("x2", x(place) - radius)
            .attr("y2", y(place));
    }
    for (const place of places) {
        const name = place.data.name;
        const details = name === "data"
            ? ["The dataset's data", ...dataDetails]
            : nodes.find(node => node.name === name).details;
        const group = svg.append("g");
        group.append("circle")
            .attr("cx", x(place)).attr("cy", y(place)).attr("r", radius)
            .append("title").text(details.join("\n"));
        group.append("text")
            .attr("class", "nodeName")
            .attr("x", x(place) - radius)
            .attr("y", y(place) - radius - 6)
            .text(name);
        if (name !== "data") { // any node but the data itself can go
            group.append("text")
                .attr("class", "nodeRemove")
                .attr("data-name", name)
                // snug on the upper right arc, below the name's line
                .attr("x", x(place) + radius - 1)
                .attr("y", y(place) - radius + 10)
                .text("×")
                .on("click", () => removeNode(name))
                .append("title").text(tip("nodeRemove"));
        }
        group.append("text")
            .attr("class", "nodeAdd")
            .attr("data-name", name)
            .attr("x", x(place) + radius - 1) // snug on the lower right arc
            .attr("y", y(place) + radius + 2)
            .text("+")
            .on("click", () => addRow(name))
            .append("title").text(tip("nodeAdd"));
    }
    graph.replaceChildren(svg.node(), note);
}

/**
 * Remove a representation and everything derived from it, felling the
 * node's whole subtree as assigning over a node does, and redraw the
 * graph, or show why it was refused
 * @param {string} name The name of the representation to remove
 */
async function removeNode(name) {
    // the node and all its descendants, gathered breadth first
    const felled = [name];
    for (const parent of felled) {
        for (const node of nodes) {
            if (node.parent === parent) felled.push(node.name);
        }
    }

    for (const target of felled) {
        const response = await fetch("/remove", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({name: target}),
        });
        if (!response.ok) {
            const reply = await response.json();
            note.textContent = `Could not remove: ${reply.error}`;
            break;
        }
        nodes = nodes.filter(node => node.name !== target);
    }
    showGraph();
    announce();
}

/**
 * Write a setting's value the way the editable fields show it
 * @param {*} value The value read from the setting's inputs
 * @returns {string} The text for a node's record of its transforms
 */
function shown(value) {
    if (Array.isArray(value)) return `[${value.join(", ")}]`;
    if (value === null) return "Random";
    return label(String(value));
}
