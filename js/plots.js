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
        itemWidth: 1, // of each item's line across the parallel axes
        markedWidth: 2, // of the marked items' lines
        shared: true, // one scale across the parallel axes, or one each
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

        // exact extents, so the plot is each feature stretched over a
        // unit square: a mark's radius is the same fraction of the plot
        // as of a transform scaled to [0, 1]
        const x = d3.scaleLinear()
            .domain(d3.extent(xValues))
            .range([margin.left, width - margin.right]);
        const y = d3.scaleLinear()
            .domain(d3.extent(yValues))
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
     * Draw a parallel coordinates chart of every feature on an axis, on one
     * shared scale written on the leftmost axis, or on its own scale when
     * shared is off; half as tall as wide unless the spec sets a height,
     * with the lines of the marked items drawn solid over the faded rest
     * @param {number[][]} data The dataset as one array of values per feature
     * @param {object} spec The width, and any defaults overridden, with
     *     features for the axis names and the indices of the items to mark
     *     as marked
     * @returns {SVGElement} The drawn chart
     */
    parallelCoordinates(data, spec = {}) {
        spec = {...this.defaults,
                margin: {...this.defaults.margin, left: 35}, ...spec};
        const {width} = spec;
        const height = spec.height ?? Math.round(width * 0.5);
        const marked = new Set(spec.marked ?? []);
        const names = spec.features ?? data.map((_, i) => String(i));
        const longest = Math.max(...names.map(name => name.length));
        const margin = {...spec.margin,
                        bottom: Math.max(spec.margin.bottom, longest * 10 + 16)};

        const x = d3.scalePoint()
            .domain(d3.range(data.length))
            .range([margin.left, width - margin.right]);
        const yTicks = Math.max(2, Math.floor(
            (height - margin.top - margin.bottom) / 40));
        const range = [height - margin.bottom, margin.top];
        let ys;
        if (spec.shared) {
            const y = d3.scaleLinear()
                .domain(d3.extent(data.flat())).nice(yTicks)
                .range(range);
            ys = data.map(() => y);
        } else {
            ys = data.map(values => d3.scaleLinear()
                .domain(d3.extent(values)).nice(yTicks)
                .range(range));
        }

        const svg = d3.create("svg")
            .attr("width", width)
            .attr("height", height);
        const ticks = group => group.style("font-family", spec.font)
            .style("font-size", spec.tickSize);

        // one line per item through its value on each feature's axis
        const line = item => d3.line()
            .x(feature => x(feature))
            .y(feature => ys[feature](data[feature][item]))
            (d3.range(data.length));
        svg.append("g")
            .selectAll("path")
            .data(d3.range(data[0].length))
            .join("path")
            .attr("d", line)
            .classed("selected", item => marked.has(item))
            .style("fill", "none")
            .style("stroke", item =>
                marked.has(item) ? spec.markedColor : spec.color)
            .style("stroke-width", item =>
                marked.has(item) ? spec.markedWidth : spec.itemWidth)
            .style("opacity", item => marked.has(item) ? 1 : spec.fade)
            .filter(item => marked.has(item))
            .raise(); // draw the marked items over the rest

        for (const [feature, y] of ys.entries()) {
            const axis = spec.shared && feature > 0
                ? d3.axisLeft(y).ticks(yTicks).tickFormat(() => "")
                : d3.axisLeft(y).ticks(yTicks);
            svg.append("g")
                .attr("transform", `translate(${x(feature)},0)`)
                .call(axis).call(ticks);
            svg.append("text")
                .attr("class", "axisLabel")
                .attr("transform", `translate(${x(feature)},` +
                    `${height - margin.bottom + 10}) rotate(-90)`)
                .attr("text-anchor", "end")
                .attr("dominant-baseline", "middle")
                .style("font-family", spec.labelFont)
                .style("font-size", spec.labelSize)
                .style("fill", "currentColor")
                .text(names[feature]);
        }

        return svg.node();
    },

    /**
     * Draw a step chart of a quantity holding its level between points,
     * such as the best loss found so far, a quarter as tall as wide
     * unless the spec sets a height
     * @param {number[][]} points The [along, value] points of the line
     * @param {object} spec The width, and any defaults overridden, with
     *     span for the x axis's reach, wholeTicks for only whole ticks
     *     written short with SI prefixes such as 10k, wholeValues for
     *     only whole ticks up the y axis, and yLabel for a name along it
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
        const yTicks = Math.max(3, Math.floor(
            (height - margin.top - margin.bottom) / 22));
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
        const yAxis = d3.axisLeft(y);
        if (spec.wholeValues) yAxis
            .tickValues(y.ticks(yTicks).filter(Number.isInteger))
            .tickFormat(d3.format("d"));
        else yAxis.ticks(yTicks);
        svg.append("g")
            .attr("transform", `translate(${margin.left},0)`)
            .call(yAxis).call(ticks);

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

        // a reference level, drawn once the chart's range reaches it
        const [low, high] = y.domain();
        if (spec.reference != null
                && spec.reference >= low && spec.reference <= high) {
            svg.append("line")
                .attr("x1", margin.left)
                .attr("x2", width - margin.right)
                .attr("y1", y(spec.reference))
                .attr("y2", y(spec.reference))
                .style("stroke", "currentColor")
                .style("stroke-opacity", 0.2);
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
