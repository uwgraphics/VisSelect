// Shared d3 chart drawing for the web application and the package's figure
// pipeline: pure functions from data and a spec to an SVG element. Styling
// rides along as inline styles, so the charts do not depend on any page
// stylesheet; the colors default to the palette's variables, defined by
// whichever page shows the chart loading palette.css

const plots = {
    // the spec every chart starts from; a call's own spec overrides it
    defaults: {
        margin: {top: 20, right: 20, bottom: 60, left: 75},
        font: "inherit", // of the tick values
        tickSize: "inherit",
        labelFont: "inherit", // of the axis names
        labelSize: "inherit",
        color: "var(--green)",
        markedColor: "var(--dark-green)",
        fade: 0.4, // the opacity of the unmarked items
        radius: 4,
        lineColor: "var(--orange)",
        lineWidth: 2,
    },

    /**
     * Draw a scatterplot of two features, square unless the spec sets a
     * height, with the marked items drawn solid over the faded rest
     * @param {number[]} xValues The values on the x axis, one per item
     * @param {number[]} yValues The values on the y axis, one per item
     * @param {object} spec The width, and any defaults overridden, with
     *     the indices of the items to mark as marked
     * @returns {SVGElement} The drawn scatterplot
     */
    scatterplot(xValues, yValues, spec = {}) {
        spec = {...this.defaults, ...spec};
        const {width, margin} = spec;
        const height = spec.height ?? width;
        const marked = new Set(spec.marked ?? []);

        const x = d3.scaleLinear()
            .domain(d3.extent(xValues)).nice()
            .range([margin.left, width - margin.right]);
        const y = d3.scaleLinear()
            .domain(d3.extent(yValues)).nice()
            .range([height - margin.bottom, margin.top]);

        const svg = d3.create("svg")
            .attr("width", width)
            .attr("height", height);
        // inline styles on the axes, over the attributes d3 sets there
        const ticks = group => group.style("font-family", spec.font)
            .style("font-size", spec.tickSize);

        svg.append("g")
            .attr("transform", `translate(0,${height - margin.bottom})`)
            .call(d3.axisBottom(x)).call(ticks);
        svg.append("g")
            .attr("transform", `translate(${margin.left},0)`)
            .call(d3.axisLeft(y)).call(ticks);

        svg.append("g")
            .selectAll("circle")
            .data(d3.range(xValues.length))
            .join("circle")
            .attr("cx", i => x(xValues[i]))
            .attr("cy", i => y(yValues[i]))
            .attr("r", spec.radius)
            .classed("selected", i => marked.has(i))
            .style("fill", i => marked.has(i) ? spec.markedColor : spec.color)
            .style("opacity", i => marked.has(i) ? 1 : spec.fade)
            .filter(i => marked.has(i))
            .raise(); // draw the marked items over the rest

        return svg.node();
    },

    /**
     * Draw a step chart of a quantity holding its level between points,
     * such as the best loss found so far, a quarter as tall as wide
     * unless the spec sets a height
     * @param {number[][]} points The [along, value] points of the line
     * @param {object} spec The width, and any defaults overridden, with
     *     span for the x axis's reach, wholeTicks for only whole ticks
     *     written short with SI prefixes such as 10k, and yLabel for a
     *     name along the y axis
     * @returns {SVGElement} The drawn chart
     */
    lossChart(points, spec = {}) {
        spec = {...this.defaults, ...spec};
        const {width, margin} = spec;
        const height = spec.height ?? Math.round(width * 0.25);
        const span = spec.span ?? points.at(-1)?.[0] ?? 1;

        // about one tick per 80 pixels across and 40 pixels up, on axes
        // rounded out to a whole tick so there is always one at each end
        const xTicks = Math.max(2, Math.floor(
            (width - margin.left - margin.right) / 80));
        const x = d3.scaleLinear()
            .domain([0, span > 0 ? span : 1])
            .range([margin.left, width - margin.right])
            .nice(xTicks);
        const yTicks = Math.max(2, Math.floor(
            (height - margin.top - margin.bottom) / 40));
        const y = d3.scaleLinear()
            .domain(points.length ? d3.extent(points, p => p[1]) : [0, 1])
            .range([height - margin.bottom, margin.top])
            .nice(yTicks);

        const svg = d3.create("svg")
            .attr("width", width)
            .attr("height", height);
        const ticks = group => group.style("font-family", spec.font)
            .style("font-size", spec.tickSize);

        const xAxis = d3.axisBottom(x);
        if (spec.wholeTicks) xAxis
            .tickValues(x.ticks(xTicks).filter(Number.isInteger))
            .tickFormat(d3.format("~s"));
        else xAxis.ticks(xTicks);
        svg.append("g")
            .attr("transform", `translate(0,${height - margin.bottom})`)
            .call(xAxis).call(ticks);
        svg.append("g")
            .attr("transform", `translate(${margin.left},0)`)
            .call(d3.axisLeft(y).ticks(yTicks)).call(ticks);

        if (spec.yLabel) {
            const middle = (margin.top + height - margin.bottom) / 2;
            svg.append("text")
                .attr("class", "axisLabel")
                .attr("transform", `translate(16,${middle}) rotate(-90)`)
                .attr("dominant-baseline", "middle")
                .attr("text-anchor", "middle")
                .style("font-family", spec.labelFont)
                .style("font-size", spec.labelSize)
                .style("fill", "currentColor")
                .text(spec.yLabel);
        }

        const line = d3.line()
            .x(point => x(point[0]))
            .y(point => y(point[1]))
            .curve(d3.curveStepAfter);
        svg.append("path")
            .attr("class", "best")
            .attr("d", line(points))
            .style("fill", "none")
            .style("stroke", spec.lineColor)
            .style("stroke-width", spec.lineWidth);

        return svg.node();
    },
};
