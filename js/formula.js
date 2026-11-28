// Objectives typeset as their terms of the loss in MathML

import {label, settingControl, settingInputs} from "./fields.js";

// each objective written as its term of the loss, by objective name
import formulas from "/json/formulas.json" with {type: "json"};

/**
 * Whether an objective's term starts with a minus, which is then written in
 * front of its weight, as − λ ∑, rather than inside the term, as λ −∑, which
 * reads as a subtraction from the weight
 * @param {string} name The objective's name in the catalog
 * @returns {boolean} Whether its term is negative
 */
export function isNegative(name) {
    return formulas[name]?.startsWith("<mo>−</mo>") ?? false;
}

/**
 * Typeset an objective as its term of the loss from its MathML template in
 * formulas.json, rendered by the browser in Libertinus Math, with each token
 * marked data-shows="setting" showing that setting's value in blue, or as its
 * name and setting fields if it has no template. Pointing at a value lays an
 * invisible control over it, since MathML cannot hold form controls: clicking
 * opens the dropdown of a setting with choices, or shows a number setting's
 * input in place, typed over the value until it is entered. A metric giving a
 * scalar turns the norm into an absolute value and drops the p of the msub
 * marked data-p.
 * @param {string} name The objective's name in the catalog
 * @param {Object} settings The objective's settings from the catalog
 * @returns {Node[]} The parts of the term in order
 */
export function formulaParts(name, settings) {
    // a negative term's leading minus is written by the loss instead
    const template = formulas[name]?.replace(/^<mo>−<\/mo>/, "");
    if (!template) return [`${label(name)}(S) `, ...settingInputs(settings)];

    const holder = document.createElement("span");
    holder.className = "math";

    // each setting's value is kept by its invisible dropdown, which is also
    // what readSettings reads when the selection is requested
    const choosers = {};
    for (const [setting, details] of Object.entries(settings)) {
        const chooser = settingControl(setting, details, details.default);
        chooser.className = "chooser";
        chooser.addEventListener("change", render);
        chooser.addEventListener("mouseleave", () => point(null));

        // a number is typed in the control itself, shown over its value
        // until it is entered or left
        if (chooser.tagName === "INPUT") {
            chooser.addEventListener("focus", () =>
                chooser.classList.add("open"));
            chooser.addEventListener("blur", () => {
                chooser.classList.remove("open");
                if (chooser.value === "") chooser.value = details.default;
                render();
            });
            chooser.addEventListener("keydown", event => {
                if (event.key === "Enter") chooser.blur();
            });
        }
        choosers[setting] = chooser;
    }

    // mark the value under the pointer, as hovering would if it were HTML
    let pointed = null;
    function point(token) {
        pointed?.classList.remove("pointed");
        pointed = token;
        token?.classList.add("pointed");
    }

    // lay a setting's dropdown over one of its values, across the value's
    // width and the whole term's height, since a value's own box is short
    function cover(chooser, token) {
        const box = token.getBoundingClientRect();
        const origin = holder.getBoundingClientRect();
        chooser.style.left = `${box.left - origin.left}px`;
        chooser.style.top = "0";
        chooser.style.width = `${box.width}px`;
        chooser.style.height = "100%";
    }

    // move a setting's dropdown onto the value the pointer entered
    function place(chooser, token) {
        cover(chooser, token);
        point(token);
    }

    function render() {
        // in display style, with full size sums and limits underneath, while
        // staying inline in its row of the loss
        holder.style.minWidth = ""; // sized again from what is drawn now
        holder.innerHTML = `<math displaystyle="true">${template}</math>`;
        const metric = choosers.metric?.value;
        if (settings.metric?.scalar.includes(metric)) {
            for (const bar of holder.querySelectorAll("mo")) {
                if (bar.textContent === "‖") bar.textContent = "|";
            }
            for (const normed of holder.querySelectorAll("msub[data-p]")) {
                normed.replaceWith(normed.firstElementChild);
            }
        }
        const shown = new Set();
        for (const token of holder.querySelectorAll("[data-shows]")) {
            const setting = token.dataset.shows;
            const chooser = choosers[setting];
            token.textContent = chooser.selectedOptions
                ? chooser.selectedOptions[0].text
                : chooser.value || chooser.placeholder;
            token.classList.add("editable");
            token.addEventListener("mouseenter", () => place(chooser, token));

            // the setting's name above its first value in the term
            if (!shown.has(chooser)) {
                const name = document.createElement("span");
                name.className = "fieldName tokenLabel";
                name.textContent = label(setting);
                name.token = token;
                holder.append(name);
            }
            shown.add(chooser);
        }
        holder.append(...shown); // only those of values still in the term
        placeNames();
    }

    // line each name up with the left of its value, above the term, and
    // keep each dropdown over the first of its values, never loose at its
    // full width over whatever follows the term
    function placeNames() {
        // Firefox sizes a term narrower than it draws its display size
        // sums, so whatever follows would overlap the term's end: hold
        // the term open to the width its math is really drawn at
        const drawn = holder.querySelector("math").getBoundingClientRect();
        holder.style.minWidth = `${drawn.width}px`;

        const origin = holder.getBoundingClientRect();
        for (const name of holder.querySelectorAll(".tokenLabel")) {
            const box = name.token.getBoundingClientRect();
            name.style.left = `${box.left - origin.left}px`;
            cover(choosers[name.token.dataset.shows], name.token);
        }
    }
    new ResizeObserver(placeNames).observe(holder);

    render();
    return [holder];
}
