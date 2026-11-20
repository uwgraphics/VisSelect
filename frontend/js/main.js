const datasetInput = document.getElementById("datasetInput");
const datasetInfo = document.getElementById("datasetInfo");
const chart = document.getElementById("chart");
const lossChart = document.getElementById("lossChart");
const featureControls = document.getElementById("featureControls");
const xFeature = document.getElementById("xFeature");
const yFeature = document.getElementById("yFeature");
const selectionControls = document.getElementById("selectionControls");
const objectives = document.getElementById("objectives");
const addObjective = document.getElementById("addObjective");
const selectSubset = document.getElementById("selectSubset");
const subsetSize = document.getElementById("subsetSize");
const algorithm = document.getElementById("algorithm");
const algorithmSettings = document.getElementById("algorithmSettings");
const selectionInfo = document.getElementById("selectionInfo");
const stopSelection = document.getElementById("stopSelection");

let data = null; // the loaded dataset as one array of values per feature
let subset = new Set(); // the indices of the items in the selected subset
let improvements = []; // the [evaluation, loss] of each new best subset
let evaluations = 1; // the number of evaluations of the selection so far

datasetInput.addEventListener("change", async () => {
    const datasetFile = datasetInput.files[0];
    if (!datasetFile) return;

    const response = await fetch("/dataset", {
        method: "POST",
        headers: {"Content-Type": "text/csv"},
        body: datasetFile,
    });
    if (!response.ok) {
        datasetInfo.textContent = "Could not load file";
        console.error(response.status, await response.text());
        return;
    }
    const reply = await response.json();
    const [rows, columns] = reply.size;
    datasetInfo.textContent = `${rows} rows × ${columns} columns`;
    data = reply.data;
    subset = new Set();
    improvements = [];
    selectionInfo.textContent = "";
    lossChart.replaceChildren();

    for (const select of [xFeature, yFeature]) {
        select.replaceChildren(...reply.features.map((name, index) =>
            new Option(name, index)));
    }
    xFeature.value = 0;
    yFeature.value = Math.min(1, columns - 1);
    featureControls.hidden = false;
    subsetSize.max = rows - 1;
    subsetSize.value = Math.min(10, rows - 1);
    selectionControls.hidden = false;
    redraw();
});

xFeature.addEventListener("change", redraw);
yFeature.addEventListener("change", redraw);

addObjective.addEventListener("change", () => {
    const row = document.createElement("div");
    row.dataset.objective = addObjective.value;

    const weight = document.createElement("input");
    weight.type = "number";
    weight.min = 0;
    weight.step = "any";
    weight.value = 1;
    const weightLabel = document.createElement("label");
    weightLabel.className = "weight";
    weightLabel.append("weight ", weight);

    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "−";
    remove.addEventListener("click", () => {
        row.remove();
        updateObjectives();
    });

    const settings = settingInputs(catalog.objectives[addObjective.value]);
    row.append(addObjective.value, " ", ...settings, weightLabel, " ", remove);
    objectives.append(row);
    addObjective.selectedIndex = 0;
    updateObjectives();
});

algorithm.addEventListener("change", showAlgorithmSettings);

selectSubset.addEventListener("click", async () => {
    const rows = [...objectives.children];
    const request = {
        objectives: rows.map(row => ({
            name: row.dataset.objective,
            weight: Number(row.querySelector(".weight input").value),
            settings: readSettings(row),
        })),
        size: Number(subsetSize.value),
        algorithm: algorithm.value,
        settings: readSettings(algorithmSettings),
    };

    selectSubset.disabled = true;
    const response = await fetch("/select", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(request),
    });
    if (!response.ok) {
        const reply = await response.json();
        selectionInfo.textContent = `Could not select a subset: ${reply.error}`;
        updateObjectives();
        return;
    }
    selectionInfo.textContent = "Selecting...";
    setRunning(true);
    improvements = [];
    evaluations = 1;
    drawLossChart(improvements, evaluations);
});

stopSelection.addEventListener("click", () => {
    fetch("/stop", {method: "POST"});
});

// follow the progress of selections the server publishes to the event stream
const events = new EventSource("/events");
events.addEventListener("message", event => {
    const message = JSON.parse(event.data);
    const ended = message.status !== "improved";
    if (ended) setRunning(false);
    if (message.status === "failed") {
        selectionInfo.textContent = `Selection failed: ${message.error}`;
        return;
    }

    subset = new Set(message.subset);
    evaluations = message.evaluation;
    if (message.status === "improved") {
        improvements.push([message.evaluation, message.loss]);
    }
    const state = {improved: " so far", stopped: " (stopped)"};
    selectionInfo.textContent = `loss ${message.loss} after ${evaluations}`
        + ` evaluations${state[message.status] ?? ""}`;
    scheduleDraw();
});

let drawPending = false; // whether a redraw is already waiting for a frame

/**
 * Redraw the charts at the next animation frame, so that a burst of
 * improvements is drawn once rather than once per message
 */
function scheduleDraw() {
    if (drawPending) return;
    drawPending = true;
    requestAnimationFrame(() => {
        drawPending = false;
        if (data) redraw();
        drawLossChart(improvements, evaluations);
    });
}

/**
 * Show the settings of the chosen algorithm
 */
function showAlgorithmSettings() {
    const settings = settingInputs(catalog.algorithms[algorithm.value]);
    algorithmSettings.replaceChildren(...settings);
}

/**
 * Create a labelled input for each setting filled with its default: a
 * dropdown for a setting with a fixed set of choices, otherwise a number
 * input where empty means None
 * @param {Object} settings Each setting's default and its choices, or null
 * @returns {HTMLLabelElement[]} The labelled inputs, one per setting
 */
function settingInputs(settings) {
    return Object.entries(settings).map(([name, {default: value, choices}]) => {
        let input;
        if (choices) {
            input = document.createElement("select");
            input.append(...choices.map(choice =>
                new Option(choice === "inf" ? "∞" : choice, choice)));
        } else {
            input = document.createElement("input");
            input.type = "number";
            input.min = 0;
            input.step = 1;
        }
        input.value = value ?? "";
        input.dataset.setting = name;
        const label = document.createElement("label");
        label.append(`${name} `, input, " ");
        return label;
    });
}

/**
 * Read the values of the setting inputs inside an element, by setting name,
 * as numbers, with empty as null and named values such as "inf" kept as text
 * @param {HTMLElement} element The element containing the setting inputs
 * @returns {Object} The value of each setting
 */
function readSettings(element) {
    const inputs = [...element.querySelectorAll("[data-setting]")];
    return Object.fromEntries(inputs.map(input => {
        const number = Number(input.value);
        const value = input.value === "" ? null
            : Number.isNaN(number) ? input.value : number;
        return [input.dataset.setting, value];
    }));
}

/**
 * Lock the dataset and every control of the problem while a selection runs,
 * leaving only Stop usable, and unlock them when it ends
 * @param {boolean} running Whether a selection is running
 */
function setRunning(running) {
    const kinds = "input, select, button";
    const controls = selectionControls.querySelectorAll(kinds);
    for (const control of [datasetInput, ...controls]) {
        control.disabled = running;
    }
    stopSelection.disabled = false;
    stopSelection.hidden = !running;
    if (!running) updateObjectives();
}

/**
 * Show weights only when blending two or more objectives, and enable
 * selecting a subset only when at least one objective has been added
 */
function updateObjectives() {
    const rows = objectives.children;
    for (const row of rows) {
        const weightLabel = row.querySelector(".weight");
        weightLabel.hidden = rows.length < 2;
        if (weightLabel.hidden) weightLabel.querySelector("input").value = 1;
    }
    selectSubset.disabled = rows.length === 0;
}

/**
 * Redraw the scatterplot with the features chosen in the dropdowns
 */
function redraw() {
    const x = Number(xFeature.value);
    const y = Number(yFeature.value);
    drawScatterplot(data, x, y, subset);
}

/**
 * Draw a scatterplot of two features of the dataset in the chart div
 * @param {number[][]} data The dataset as one array of values per feature
 * @param {number} xFeature The index of the feature on the x axis
 * @param {number} yFeature The index of the feature on the y axis
 * @param {Set<number>} subset The indices of the selected items to mark
 */
function drawScatterplot(data, xFeature, yFeature, subset) {
    const width = 600;
    const height = 600;
    const margin = 40;
    const xValues = data[xFeature];
    const yValues = data[yFeature];

    const x = d3.scaleLinear()
        .domain(d3.extent(xValues)).nice()
        .range([margin, width - margin]);

    const y = d3.scaleLinear()
        .domain(d3.extent(yValues)).nice()
        .range([height - margin, margin]);

    const svg = d3.create("svg")
        .attr("width", width)
        .attr("height", height);

    svg.append("g")
        .attr("transform", `translate(0,${height - margin})`)
        .call(d3.axisBottom(x));

    svg.append("g")
        .attr("transform", `translate(${margin},0)`)
        .call(d3.axisLeft(y));

    svg.append("g")
        .selectAll("circle")
        .data(d3.range(xValues.length))
        .join("circle")
        .attr("cx", i => x(xValues[i]))
        .attr("cy", i => y(yValues[i]))
        .attr("r", 4)
        .classed("selected", i => subset.has(i));

    chart.replaceChildren(svg.node());
}

/**
 * Draw a step chart of the best loss found so far over the evaluations
 * @param {number[][]} improvements The [evaluation, loss] of each new best
 * @param {number} evaluations The number of evaluations so far
 */
function drawLossChart(improvements, evaluations) {
    const width = 400;
    const height = 600;
    const margin = 40;

    // hold the latest best loss level out to the latest evaluation
    const last = improvements.at(-1);
    const points = last ? [...improvements, [evaluations, last[1]]] : [];

    const x = d3.scaleLinear()
        .domain([1, Math.max(evaluations, 2)])
        .range([margin, width - margin]);

    const y = d3.scaleLinear()
        .domain(points.length ? d3.extent(points, p => p[1]) : [0, 1]).nice()
        .range([height - margin, margin]);

    const line = d3.line()
        .x(p => x(p[0]))
        .y(p => y(p[1]))
        .curve(d3.curveStepAfter);

    const svg = d3.create("svg")
        .attr("width", width)
        .attr("height", height);

    svg.append("g")
        .attr("transform", `translate(0,${height - margin})`)
        .call(d3.axisBottom(x));

    svg.append("g")
        .attr("transform", `translate(${margin},0)`)
        .call(d3.axisLeft(y));

    svg.append("path") // the best loss found so far
        .attr("d", line(points))
        .attr("class", "best");

    lossChart.replaceChildren(svg.node());
}

// fill the objective and algorithm dropdowns from the package's catalog
const catalog = await (await fetch("/catalog")).json();
const objectiveNames = Object.keys(catalog.objectives);
addObjective.append(...objectiveNames.map(name => new Option(name)));
const algorithmNames = Object.keys(catalog.algorithms);
algorithm.append(...algorithmNames.map(name => new Option(name)));
showAlgorithmSettings();
