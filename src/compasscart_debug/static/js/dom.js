/** Small DOM helpers shared by the debug console. */

export function createElement(tagName, attributes = {}, children = []) {
  const node = document.createElement(tagName);
  for (const [name, value] of Object.entries(attributes || {})) {
    if (value === undefined || value === null || value === false) continue;
    if (name === "className") {
      node.className = String(value);
    } else if (name === "textContent") {
      node.textContent = String(value);
    } else if (name === "dataset" && value && typeof value === "object") {
      for (const [key, item] of Object.entries(value)) node.dataset[key] = String(item);
    } else if (name === "on" && value && typeof value === "object") {
      for (const [eventName, handler] of Object.entries(value)) {
        if (typeof handler === "function") node.addEventListener(eventName, handler);
      }
    } else if (name === "checked" || name === "disabled" || name === "readOnly" || name === "open") {
      node[name] = Boolean(value);
    } else if (name === "htmlFor") {
      node.htmlFor = String(value);
    } else {
      node.setAttribute(name, String(value));
    }
  }
  append(node, children);
  return node;
}

export const el = createElement;

export function text(value) {
  return document.createTextNode(value === null || value === undefined ? "" : String(value));
}

export function append(parent, children) {
  const values = Array.isArray(children) ? children : [children];
  for (const child of values) {
    if (child === null || child === undefined || child === false) continue;
    parent.append(child instanceof Node ? child : text(child));
  }
  return parent;
}

export function clear(node) {
  while (node && node.firstChild) node.removeChild(node.firstChild);
  return node;
}

export function nullable(value, missing = "未记录") {
  if (value === null || value === undefined || value === "") return missing;
  return String(value);
}

export function formatNumber(value, missing = "未记录") {
  if (value === null || value === undefined || value === "" || Number.isNaN(Number(value))) return missing;
  return new Intl.NumberFormat("en-US").format(Number(value));
}

export function formatPrice(value) {
  if (value === null || value === undefined || value === "" || Number.isNaN(Number(value))) return "价格未记录";
  const numeric = Number(value);
  return Number.isFinite(numeric) ? `$${numeric.toFixed(2)}` : "价格未记录";
}

export function prettyJson(value) {
  if (value === null || value === undefined) return "未记录";
  try {
    const serialized = JSON.stringify(value, null, 2);
    return serialized === undefined ? "未记录" : serialized;
  } catch {
    return "无法显示";
  }
}

export function setVisible(node, visible) {
  if (node) node.hidden = !visible;
  return node;
}

export function compactList(value, missing = "未记录") {
  if (!Array.isArray(value) || value.length === 0) return missing;
  return value.map((item) => nullable(item)).join("、");
}

export function labelValue(label, value, className = "meta-row") {
  return createElement("div", { className }, [
    createElement("dt", { className: "meta-label", textContent: label }),
    createElement("dd", { className: "meta-value", textContent: nullable(value) }),
  ]);
}

