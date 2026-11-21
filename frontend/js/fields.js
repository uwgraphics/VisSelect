// Fields showing settings and values, each with its name above its value

// the names shown on the page for package names, by package name
import labels from "/json/labels.json" with {type: "json"};
// the tooltips of the page's own controls, by element id or control name
import tooltips from "/json/tooltips.json" with {type: "json"};

let nextSettingId = 0; // numbers the setting inputs so labels can name them

/**
 * Look up the name shown on the page for a package name, which is the package
 * name itself until it is given a label in labels.json
 * @param {string} name The name of a generator, setting or other package entry
 * @returns {string} The label to show for it
 */
export function label(name) {
    return labels[name] ?? name;
}

/**
 * Look up the tooltip of one of the page's own controls in tooltips.json;
 * settings instead carry their descriptions from the package's docstrings
 * @param {string} name The element id or control name
 * @returns {string} The tooltip, empty for none
 */
export function tip(name) {
    return tooltips[name] ?? "";
}

/**
 * Create a dropdown option for each package name, showing its label
 * @param {string[]} names The package names, which are the options' values
 * @returns {HTMLOptionElement[]} The options
 */
export function options(names) {
    return names.map(name => new Option(label(name), name));
}

/**
 * Create a labelled input for each setting filled with its default: a
 * dropdown for a setting with a fixed set of choices, otherwise a number
 * input, whole and at least 0 or decimal, where empty means None, with two
 * inputs for a setting that takes a pair
 * @param {Object} settings Each setting's default, its choices or null, and
 *     whether it is an integer and whether it is a pair
 * @returns {HTMLSpanElement[]} The labelled inputs, one field per setting
 */
export function settingInputs(settings) {
    return Object.entries(settings).map(([name, setting]) => {
        const {default: value, pair} = setting;
        const values = pair ? value ?? [null, null] : [value];
        const inputs = values.map(value =>
            fitted(settingControl(name, setting, value)));

        // the name above the values, with the two values of a size as
        // rows × columns, of the data's interval as a plain range, and of
        // other pairs as a fixed value or a range to draw from
        const fieldName = document.createElement("label");
        fieldName.className = "fieldName";
        fieldName.textContent = label(name);
        if (setting.description) fieldName.title = setting.description;
        const fieldValue = document.createElement("span");
        fieldValue.className = "fieldValue";
        if (name === "size") {
            const times = document.createElement("span");
            times.className = "times";
            times.textContent = "×";
            fieldValue.append(inputs[0], times, inputs[1]);
        } else if (name === "interval") { // the range of the data itself
            fieldValue.append("[", inputs[0], ", ", inputs[1], "]");
        } else if (pair) {
            fieldValue.append(...rangeParts(inputs));
        } else {
            fieldValue.append(...inputs);
        }
        const field = document.createElement("span");
        field.className = "field";
        field.append(fieldName, " ", fieldValue, " ");

        // only the name is the label, of the first value, since hovering a
        // label also hovers its input, which would grey the first value
        // whenever the pointer is over the second
        const first = field.querySelector("[data-setting]");
        first.id = `setting${nextSettingId++}`;
        fieldName.htmlFor = first.id;
        return field;
    });
}

/**
 * Create a field showing a value that cannot be edited, with its name above
 * it in the same style as the settings
 * @param {string} name The name shown above the value
 * @param {string|number} value The value
 * @returns {HTMLSpanElement} The field
 */
export function infoField(name, value) {
    const fieldName = document.createElement("span");
    fieldName.className = "fieldName";
    fieldName.textContent = name;
    const fieldValue = document.createElement("span");
    fieldValue.className = "fieldValue";
    fieldValue.textContent = value;
    const field = document.createElement("span");
    field.className = "field";
    field.append(fieldName, " ", fieldValue, " ");
    return field;
}

/**
 * Create the input for one value of a setting: a dropdown for a setting with
 * a fixed set of choices, otherwise a number input, whole and at least 0 or
 * decimal, where empty means None
 * @param {string} name The setting's name
 * @param {Object} setting The setting's default, choices, and whether it is
 *     an integer and whether it is a pair
 * @param {*} value The value to start with
 * @returns {HTMLInputElement|HTMLSelectElement} The input
 */
export function settingControl(name, setting, value) {
    const {choices, integer, pair} = setting;
    let input;
    if (choices) {
        input = document.createElement("select");
        input.append(...choices.map(choice =>
            new Option(choice === "inf" ? "∞" : label(choice), choice)));
    } else {
        input = document.createElement("input");
        input.type = "number";
        input.step = integer ? 1 : "any";
        // a dataset needs at least one item and one feature
        if (integer) input.min = name === "size" ? 1 : 0;
    }
    input.classList.add("editable");
    input.value = value ?? "";
    input.dataset.setting = name;
    if (setting.description) input.title = setting.description;

    // an empty setting is sent as None, which means random for seeds
    if (value == null && !choices) input.placeholder = "Random";

    if (pair) input.dataset.pair = "";
    return input;
}

/**
 * Lay out the two inputs of a range setting behind a switch between a fixed
 * value, shown as "= v" and sent as [v, v], and a range to draw from
 * uniformly at random, shown as "∼ [low, high]", starting fixed when its two
 * bounds are equal
 * @param {HTMLElement[]} bounds The fitted inputs of the low and high bounds
 * @returns {HTMLElement[]} The parts of the field in order
 */
function rangeParts([low, high]) {
    const lowInput = low.querySelector("input");
    const highInput = high.querySelector("input");
    const part = text => {
        const span = document.createElement("span");
        span.textContent = text;
        return span;
    };
    const open = part("[");
    const comma = part(", ");
    const close = part("]");
    const rangeOnly = [open, comma, high, close];

    const relation = document.createElement("button");
    relation.type = "button";
    relation.className = "relation editable";

    // while fixed, the high bound follows the low one so [v, v] is sent
    const follow = () => {
        highInput.value = lowInput.value;
        highInput.dispatchEvent(new Event("input")); // refit it
    };
    const show = random => {
        relation.textContent = random ? "∼" : "=";
        relation.title = random ? tip("relationRandom") : tip("relationFixed");
        for (const element of rangeOnly) element.hidden = !random;
        if (!random) follow();
    };
    const isRandom = () => !open.hidden;

    lowInput.addEventListener("input", () => {
        if (!isRandom()) follow();
    });
    relation.addEventListener("click", () => {
        show(!isRandom());
        if (isRandom()) { // ready to type the high bound over the low one
            highInput.focus();
            highInput.select();
        }
    });
    show(lowInput.value !== highInput.value);
    return [relation, open, low, comma, high, close];
}

/**
 * Wrap an input or dropdown in a span that keeps a copy of the text it shows,
 * its value, its placeholder when empty, or its chosen option, for styles
 * that fit the input to its text
 * @param {HTMLInputElement|HTMLSelectElement} input The input to wrap
 * @returns {HTMLSpanElement} The wrapper, in the input's place if it had one
 */
export function fitted(input) {
    const wrapper = document.createElement("span");
    wrapper.className = "fit";
    const copy = () => {
        const shown = input.tagName === "SELECT"
            ? input.selectedOptions[0]?.text
            : input.value || input.placeholder;
        wrapper.dataset.value = shown ?? "";
    };
    copy();
    input.addEventListener("input", copy);
    input.addEventListener("change", copy);
    input.before(wrapper); // does nothing if the input is not on the page
    wrapper.append(input);
    return wrapper;
}

/**
 * Read the values of the setting inputs inside an element, by setting name,
 * as numbers, with empty as null and named values such as "inf" kept as text,
 * and the two values of a pair setting together in an array
 * @param {HTMLElement} element The element containing the setting inputs
 * @returns {Object} The value of each setting
 */
export function readSettings(element) {
    const settings = {};
    for (const input of element.querySelectorAll("[data-setting]")) {
        const name = input.dataset.setting;
        const number = Number(input.value);
        const value = input.value === "" ? null
            : Number.isNaN(number) ? input.value : number;
        if ("pair" in input.dataset) (settings[name] ??= []).push(value);
        else settings[name] = value;
    }
    return settings;
}

/**
 * Lock or unlock every input, dropdown and small switch inside an element,
 * marking it as running so the styles show its values as fixed
 * @param {HTMLElement} element The element containing the controls
 * @param {boolean} locked Whether to lock them
 */
export function lockControls(element, locked) {
    const controls = "input, select, button:not(.button)";
    for (const control of element.querySelectorAll(controls)) {
        control.disabled = locked;
    }
    element.classList.toggle("running", locked);
}
