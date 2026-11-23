// The scatterplot of the dataset and the line chart of the loss

import {fitted} from "./fields.js";
import {representationNames} from "./transforms.js";

const chart = document.getElementById("chart");
const lossChart = document.getElementById("lossChart");
const plot = document.getElementById("plot");
const xFeature = document.getElementById("xFeature");
const yFeature = document.getElementById("yFeature");
const xAxisField = document.getElementById("xAxisField");
const yAxisField = document.getElementById("yAxisField");
const chartType = document.getElementById("chartType");
const chartOn = document.getElementById("chartOn");
const chartOnField = document.getElementById("chartOnField");
const lossXAxis = document.getElementById("lossXAxis");
const lossXAxisField = document.getElementById("lossXAxisField");

// what the charts show, kept to redraw them when their space changes size
let data = null; // the loaded dataset as one array of values per feature
let features = []; // the feature names of what the scatterplot shows
let subset = new Set(); // the indices of the items in the selected subset
let improvements = []; // the [evaluation, time, loss] of each new best subset
let evaluations = 1; // the number of evaluations of the selection so far
let elapsed = 0; // the seconds the selection has run so far

let drawPending = false; // whether a redraw is already waiting for a frame

// the charts styles
const styling = {
    font: "var(--page-font)",
    tickSize: "var(--tick-size)",
    labelFont: "var(--control-font)",
    labelSize: "var(--control-size)",
};

/**
 * Set up the feature dropdowns on the scatterplot's axes and redraw the
 * charts whenever their space on the page changes size
 */
export function setupCharts() {
    fitted(xFeature);
    fitted(yFeature);
    xFeature.addEventListener("change", redraw);
    yFeature.addEventListener("change", redraw);

    // which representation the chart shows, offered once there is more
    // than one, following the transform tree as it changes
    fitted(chartType);
    chartOn.append(new Option("data"));
    fitted(chartOn);
    chartOn.addEventListener("change", showRepresentation);
    document.addEventListener("representationsChanged", () => {
        const names = representationNames();
        chartOnField.hidden = names.length < 2;
        const value = names.includes(chartOn.value) ? chartOn.value : "data";
        chartOn.replaceChildren(...names.map(name => new Option(name)));
        chartOn.value = value;
        // refetch what is shown: it may have just been transformed over
        chartOn.dispatchEvent(new Event("change"));
    });

    // the loss chart's x axis toggles between evaluations and time
    fitted(lossXAxis);
    lossXAxis.addEventListener("change", drawLossChart);

    const resizing = new ResizeObserver(scheduleDraw);
    resizing.observe(chart);
    resizing.observe(lossChart);
}

/**
 * Show the representation chosen for the chart, fetched from the server,
 * keeping the marked subset, whose items are the same rows everywhere, and
 * the chosen axes when the features are unchanged
 */
async function showRepresentation() {
    const response = await fetch(
        `/representation?name=${encodeURIComponent(chartOn.value)}`);
    if (!response.ok) return;
    const reply = await response.json();
    if (reply.features.join("\u0000") !== features.join("\u0000")) {
        showFeatures(reply.features);
    }
    data = reply.data;
    redraw();
}

/**
 * Fill the axes' feature dropdowns for a newly loaded dataset, with the
 * first two features on the axes, and show the scatterplot
 * @param {string[]} features The dataset's feature names
 */
export function showFeatures(names) {
    features = names;
    // features named only by their column numbers, as generated datasets are,
    // shown as "Feature 0" and so on so a bare number reads as a feature
    const numbered = names.every((name, i) => name === String(i));
    for (const select of [xFeature, yFeature]) {
        select.replaceChildren(...names.map((name, index) =>
            new Option(numbered ? `Feature ${name}` : name, index)));
    }
    xFeature.value = 0;
    yFeature.value = Math.min(1, names.length - 1);
    for (const select of [xFeature, yFeature]) {
        select.dispatchEvent(new Event("input")); // refit it
    }
    plot.hidden = false;
}

/**
 * Show a newly loaded dataset on the scatterplot, with no subset selected
 * @param {number[][]} values The dataset as one array of values per feature
 */
export function showData(values) {
    data = values;
    subset = new Set();
    redraw();
}

/**
 * Mark the best subset found so far on the scatterplot
 * @param {Set<number>} indices The indices of the items in the subset
 */
export function showSubset(indices) {
    subset = indices;
    scheduleDraw();
}

/**
 * Start a fresh loss chart for a selection that has just started
 */
export function startLoss() {
    improvements = [];
    evaluations = 1;
    elapsed = 0;
    drawLossChart();
}

/**
 * Show the best loss found so far over the evaluations on the loss chart
 * @param {number[][]} best The [evaluation, time, loss] of each new best
 *     subset
 * @param {number} count The number of evaluations so far
 * @param {number} time The seconds the selection has run so far
 */
export function showLoss(best, count, time) {
    improvements = best;
    evaluations = count;
    elapsed = time;
    scheduleDraw();
}

/**
 * Redraw the charts at the next animation frame, so that a burst of
 * improvements is drawn once rather than once per message
 */
export function scheduleDraw() {
    if (drawPending) return;
    drawPending = true;
    requestAnimationFrame(() => {
        drawPending = false;
        if (data) redraw();
        // only once a selection has started drawing it
        if (lossChart.hasChildNodes()) drawLossChart();
    });
}

/**
 * Redraw the scatterplot with the features chosen in the dropdowns
 */
export function redraw() {
    const x = Number(xFeature.value);
    const y = Number(yFeature.value);
    drawScatterplot(data, x, y, subset);
}

/**
 * Remove the loss chart, until a selection starts drawing it again
 */
export function clearLossChart() {
    improvements = [];
    evaluations = 1;
    elapsed = 0;
    lossChart.replaceChildren();
    lossXAxisField.hidden = true;
}

/**
 * Draw a square scatterplot of two features of the dataset filling the width
 * of the chart div
 * @param {number[][]} data The dataset as one array of values per feature
 * @param {number} xIndex The index of the feature on the x axis
 * @param {number} yIndex The index of the feature on the y axis
 * @param {Set<number>} subset The indices of the selected items to mark
 */
function drawScatterplot(data, xIndex, yIndex, subset) {
    const width = chart.clientWidth;
    const margin = plots.defaults.margin;

    // the feature dropdowns centred along their axes, as axis names
    xAxisField.style.left = `${(margin.left + width - margin.right) / 2}px`;
    yAxisField.style.top = `${(margin.top + width - margin.bottom) / 2}px`;

    chart.replaceChildren(plots.scatterplot(data[xIndex], data[yIndex],
        {...styling, width, marked: subset}));
}

/**
 * Draw a step chart of the best loss found so far over the evaluations,
 * four times as wide as tall, filling the width of the loss chart div
 */
function drawLossChart() {
    const width = lossChart.clientWidth;

    // the x axis counts evaluations or seconds, as toggled on the axis;
    // whole evaluations, while seconds keep their decimals
    const timed = lossXAxis.value === "time";
    const along = timed ? point => point[1] : point => point[0];
    const span = timed ? elapsed : evaluations;

    // hold the latest best loss level out to the latest progress
    const last = improvements.at(-1);
    const points = last
        ? [...improvements.map(point => [along(point), point[2]]),
           [span, last[2]]]
        : [];

    // the y axis name goes along the axis; the x axis name is the toggle,
    // centred below the axis like the scatterplot's feature dropdowns
    const margin = plots.defaults.margin;
    lossXAxisField.style.left =
        `${(margin.left + width - margin.right) / 2}px`;
    lossXAxisField.hidden = false;

    lossChart.replaceChildren(plots.lossChart(points,
        {...styling, width, span, wholeTicks: !timed, yLabel: "Loss"}));
}
