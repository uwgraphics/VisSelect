// The scatterplot of the dataset and the line chart of the loss

import {fitted} from "./fields.js";

const chart = document.getElementById("chart");
const lossChart = document.getElementById("lossChart");
const plot = document.getElementById("plot");
const xFeature = document.getElementById("xFeature");
const yFeature = document.getElementById("yFeature");
const xAxisField = document.getElementById("xAxisField");
const yAxisField = document.getElementById("yAxisField");

// what the charts show, kept to redraw them when their space changes size
let data = null; // the loaded dataset as one array of values per feature
let subset = new Set(); // the indices of the items in the selected subset
let improvements = []; // the [evaluation, loss] of each new best subset
let evaluations = 1; // the number of evaluations of the selection so far

let drawPending = false; // whether a redraw is already waiting for a frame

/**
 * Set up the feature dropdowns on the scatterplot's axes and redraw the
 * charts whenever their space on the page changes size
 */
export function setupCharts() {
    fitted(xFeature);
    fitted(yFeature);
    xFeature.addEventListener("change", redraw);
    yFeature.addEventListener("change", redraw);

    const resizing = new ResizeObserver(scheduleDraw);
    resizing.observe(chart);
    resizing.observe(lossChart);
}

/**
 * Fill the axes' feature dropdowns for a newly loaded dataset, with the
 * first two features on the axes, and show the scatterplot
 * @param {string[]} features The dataset's feature names
 */
export function showFeatures(features) {
    // features named only by their column numbers, as generated datasets are,
    // shown as "Feature 0" and so on so a bare number reads as a feature
    const numbered = features.every((name, i) => name === String(i));
    for (const select of [xFeature, yFeature]) {
        select.replaceChildren(...features.map((name, index) =>
            new Option(numbered ? `Feature ${name}` : name, index)));
    }
    xFeature.value = 0;
    yFeature.value = Math.min(1, features.length - 1);
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
    drawLossChart();
}

/**
 * Show the best loss found so far over the evaluations on the loss chart
 * @param {number[][]} best The [evaluation, loss] of each new best subset
 * @param {number} count The number of evaluations so far
 */
export function showLoss(best, count) {
    improvements = best;
    evaluations = count;
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
    lossChart.replaceChildren();
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
    const height = width;
    // room on the left and bottom for the tick values and feature dropdowns
    const margin = {top: 20, right: 20, bottom: 60, left: 75};
    const xValues = data[xIndex];
    const yValues = data[yIndex];

    const x = d3.scaleLinear()
        .domain(d3.extent(xValues)).nice()
        .range([margin.left, width - margin.right]);

    const y = d3.scaleLinear()
        .domain(d3.extent(yValues)).nice()
        .range([height - margin.bottom, margin.top]);

    const svg = d3.create("svg")
        .attr("width", width)
        .attr("height", height);

    svg.append("g")
        .attr("transform", `translate(0,${height - margin.bottom})`)
        .call(d3.axisBottom(x));

    svg.append("g")
        .attr("transform", `translate(${margin.left},0)`)
        .call(d3.axisLeft(y));

    // the feature dropdowns centred along their axes, as axis names
    xAxisField.style.left = `${(margin.left + width - margin.right) / 2}px`;
    yAxisField.style.top = `${(margin.top + height - margin.bottom) / 2}px`;

    svg.append("g")
        .selectAll("circle")
        .data(d3.range(xValues.length))
        .join("circle")
        .attr("cx", i => x(xValues[i]))
        .attr("cy", i => y(yValues[i]))
        .attr("r", 4)
        .classed("selected", i => subset.has(i))
        .filter(i => subset.has(i))
        .raise(); // draw the subset over the rest of the dataset

    chart.replaceChildren(svg.node());
}

/**
 * Draw a step chart of the best loss found so far over the evaluations,
 * four times as wide as tall, filling the width of the loss chart div
 */
function drawLossChart() {
    const width = lossChart.clientWidth;
    const height = Math.round(width * 0.25); // four times as wide as tall
    // room on the left and bottom for the tick values and axis names
    const margin = {top: 20, right: 20, bottom: 60, left: 75};

    // hold the latest best loss level out to the latest evaluation
    const last = improvements.at(-1);
    const points = last ? [...improvements, [evaluations, last[1]]] : [];

    // evaluation counts get long, so about one tick per 80 pixels, on an axis
    // from 0 rounded out to a whole tick so there is always one at each end
    const xTicks = Math.max(2, Math.floor(
        (width - margin.left - margin.right) / 80));
    const x = d3.scaleLinear()
        .domain([0, Math.max(evaluations, 1)])
        .range([margin.left, width - margin.right])
        .nice(xTicks);

    // about one tick per 40 pixels of height, rounded out to a tick at each end
    const yTicks = Math.max(2, Math.floor(
        (height - margin.top - margin.bottom) / 40));
    const y = d3.scaleLinear()
        .domain(points.length ? d3.extent(points, p => p[1]) : [0, 1])
        .range([height - margin.bottom, margin.top])
        .nice(yTicks);

    const line = d3.line()
        .x(p => x(p[0]))
        .y(p => y(p[1]))
        .curve(d3.curveStepAfter);

    const svg = d3.create("svg")
        .attr("width", width)
        .attr("height", height);

    // only whole evaluations, written short with SI prefixes such as 10k
    const xAxis = d3.axisBottom(x)
        .tickValues(x.ticks(xTicks).filter(Number.isInteger))
        .tickFormat(d3.format("~s"));
    svg.append("g")
        .attr("transform", `translate(0,${height - margin.bottom})`)
        .call(xAxis);

    svg.append("g")
        .attr("transform", `translate(${margin.left},0)`)
        .call(d3.axisLeft(y).ticks(yTicks));

    // the axis names, centred along each axis at the edge of the chart
    svg.append("text")
        .attr("class", "axisLabel")
        .attr("x", (margin.left + width - margin.right) / 2)
        .attr("y", height - 8)
        .attr("text-anchor", "middle")
        .text("Evaluations");

    const middle = (margin.top + height - margin.bottom) / 2;
    svg.append("text")
        .attr("class", "axisLabel")
        .attr("transform", `translate(16,${middle}) rotate(-90)`)
        .attr("dominant-baseline", "middle")
        .attr("text-anchor", "middle")
        .text("Loss");

    svg.append("path") // the best loss found so far
        .attr("d", line(points))
        .attr("class", "best");

    lossChart.replaceChildren(svg.node());
}
