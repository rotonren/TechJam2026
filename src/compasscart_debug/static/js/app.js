import { createApi, downloadBlob, exportBlob, readImportFile } from "./api.js";
import { append, clear, compactList, createElement, formatNumber, formatPrice, nullable, prettyJson, setVisible } from "./dom.js";
import { createStore } from "./store.js";

const FEEDBACK_REASONS = [
  ["explicit_constraint", "违反明确约束"],
  ["wrong_category", "错误品类"],
  ["over_budget", "预算不符"],
  ["attribute_mismatch", "属性不符"],
  ["duplicate_or_too_similar", "重复或过于相似"],
  ["other", "其他"],
];

const MOBILE_TABS = ["conversation", "recommendations", "inspector"];

export function shouldSubmitComposerEvent(event) {
  if (!event || event.key !== "Enter" || event.shiftKey) return false;
  return !Boolean(event.isComposing || event.nativeEvent?.isComposing || event.keyCode === 229);
}

function byId(id) {
  return typeof document === "undefined" ? null : document.getElementById(id);
}

function isElement(value) {
  return typeof Element !== "undefined" && value instanceof Element;
}

function captureFocus() {
  const active = typeof document !== "undefined" ? document.activeElement : null;
  if (!active || !active.id) return null;
  return { id: active.id, selectionStart: active.selectionStart, selectionEnd: active.selectionEnd };
}

function restoreFocus(snapshot) {
  if (!snapshot) return;
  const node = byId(snapshot.id);
  if (!node || typeof node.focus !== "function") return;
  node.focus({ preventScroll: true });
  if (typeof node.setSelectionRange === "function" && snapshot.selectionStart !== undefined) {
    try { node.setSelectionRange(snapshot.selectionStart, snapshot.selectionEnd); } catch { /* non-text controls */ }
  }
}

function button(label, attrs = {}) {
  return createElement("button", { className: "button button-secondary", type: "button", ...attrs }, [label]);
}

function statusClass(state) {
  if (state.loginError) return "is-error";
  if (state.busy) return "is-busy";
  if (state.serviceStatus === "ready") return "is-ready";
  return "";
}

function displayStatus(state) {
  if (state.loginError) return "需要重新连接";
  if (state.busy) return "处理中";
  if (state.serviceStatus === "ready") return "Agent 在线";
  if (state.serviceStatus === "checking") return "连接中";
  return "未连接";
}

function normalizeTags(value) {
  if (Array.isArray(value)) return value.map((item) => String(item)).filter(Boolean);
  if (typeof value === "string") return value.split(",").map((item) => item.trim()).filter(Boolean);
  return [];
}

function formatDetail(value) {
  if (Array.isArray(value)) return compactList(value);
  if (value && typeof value === "object") return prettyJson(value);
  return nullable(value);
}

function feedbackFor(product, turn) {
  return (Array.isArray(turn?.feedback) ? turn.feedback : []).find((item) => item.parent_asin === product.parent_asin) || null;
}

function renderSessionControls(state) {
  const select = byId("session-select");
  const list = byId("session-list");
  if (!select || !list) return;
  clear(select);
  if (!state.sessions.length) {
    select.append(createElement("option", { value: "", textContent: "暂无会话" }));
  } else {
    for (const session of state.sessions) {
      select.append(createElement("option", {
        value: session.session_id,
        textContent: session.archived ? `${session.name} · 已归档` : session.name,
        selected: session.session_id === state.currentSession?.session_id,
      }));
    }
  }
  clear(list);
  for (const session of state.sessions) {
    list.append(createElement("button", {
      className: `session-chip${session.session_id === state.currentSession?.session_id ? " is-active" : ""}`,
      type: "button",
      title: session.name,
      "aria-pressed": session.session_id === state.currentSession?.session_id,
      dataset: { sessionId: session.session_id },
    }, [session.name || "未命名会话"]));
  }
}

function renderConversation(state) {
  const history = byId("conversation-history");
  const empty = byId("empty-conversation");
  const composer = byId("message-composer");
  const input = byId("message-input");
  const send = byId("send-button");
  const limitActions = byId("turn-limit-actions");
  if (!history || !empty || !composer || !input || !send || !limitActions) return;
  clear(history);
  const turns = state.turns || [];
  setVisible(empty, !state.currentSession || turns.length === 0);
  for (const turn of turns) history.append(renderTurn(turn, state));
  const limited = state.continuation === "turn_limit" || turns.filter((turn) => turn.status === "completed").length >= 10;
  const blocked = !state.currentSession || Boolean(state.currentSession.archived) || !state.canSend || limited || state.busy;
  input.disabled = blocked;
  send.disabled = blocked;
  setVisible(composer, !limited);
  setVisible(limitActions, limited);
  // The preference field belongs to the new-session dialog, not the active
  // conversation. Keep it editable even when the current session has turns.
  const tagInput = byId("preference-tags");
  if (tagInput) tagInput.readOnly = false;
  if (state.currentSession?.archived) input.disabled = true;
  const hint = byId("composer-hint");
  if (hint) hint.textContent = state.currentSession?.archived ? "该会话已归档" : (limited ? "" : "Enter 发送 · Shift+Enter 换行");
}

function renderTurn(turn, state) {
  const selected = Number(turn.turn) === Number(state.selectedTurn);
  const article = createElement("article", { className: `turn-entry${selected ? " is-selected" : ""}` });
  const marker = createElement("div", { className: "turn-marker", textContent: `第 ${turn.turn} 轮 · ${turn.status === "completed" ? "完成" : turn.status === "pending" ? "处理中" : "失败"}` });
  const select = createElement("button", { className: "turn-select", type: "button", dataset: { turn: turn.turn }, "aria-label": `选择第 ${turn.turn} 轮` });
  append(select, [
    createElement("div", { className: "message message-user" }, [createElement("span", { className: "message-label", textContent: "USER" }), turn.user_message || ""]),
  ]);
  if (turn.status === "completed") {
    const response = turn.response || {};
    append(select, [createElement("div", { className: "message message-agent" }, [createElement("span", { className: "message-label", textContent: "AGENT" }), response.message || "未记录 Agent 回复"]) ]);
    if (response.ask_attribute) append(select, [createElement("div", { className: "ask-attribute" }, [`需要补充：${response.ask_attribute}`])]);
  } else if (turn.status === "pending") {
    append(select, [createElement("div", { className: "turn-pending" }, ["等待 Agent 响应…"])]);
  } else {
    const retry = button("重试", { className: "button button-secondary", dataset: { retryTurn: turn.turn } });
    article.append(marker, select, createElement("div", { className: "turn-failed" }, [createElement("span", { textContent: turn.error?.message || "本轮失败，可重试" }), retry]));
    return article;
  }
  article.append(marker, select);
  return article;
}

function renderRecommendations(state, ui) {
  const list = byId("recommendations-list");
  const empty = byId("empty-recommendations");
  const count = byId("recommendation-count");
  const context = byId("recommendation-context");
  if (!list || !empty || !count || !context) return;
  const observation = state.selectedObservation;
  const products = Array.isArray(observation?.products) ? observation.products : [];
  count.textContent = String(products.length);
  context.textContent = observation ? `第 ${observation.turn} 轮 · 严格按 Agent 响应顺序` : "等待选择一轮";
  clear(list);
  setVisible(empty, !observation);
  for (const product of products) list.append(renderProduct(product, observation, ui));
}

function renderProduct(product, turn, ui) {
  const row = createElement("article", { className: "product-row" });
  const existing = feedbackFor(product, turn);
  const key = `${turn.turn}:${product.parent_asin}`;
  const expanded = ui.feedbackKey === key;
  const title = product.metadata_missing ? "商品元数据缺失" : nullable(product.title, "标题未记录");
  const main = createElement("div", { className: "product-main" });
  main.append(
    createElement("span", { className: "rank", textContent: String(product.rank ?? "?") }),
    createElement("div", { className: "product-title-block" }, [
      createElement("h3", { className: "product-title", textContent: title }),
      createElement("div", { className: "product-asin", textContent: `ASIN ${nullable(product.parent_asin)}` }),
    ]),
    createElement("div", { className: "product-actions" }, [createElement("button", {
      className: "feedback-toggle",
      type: "button",
      title: expanded ? "收起反馈" : "标记商品不准确",
      "aria-label": expanded ? "收起反馈" : "标记商品不准确",
      "aria-expanded": expanded,
      dataset: { feedbackToggle: key },
      textContent: existing ? "!" : "⚑",
    })]),
  );
  const meta = createElement("div", { className: "product-meta" });
  const fields = [
    ["价格", formatPrice(product.price)],
    ["评分", nullable(product.rating)],
    ["评分数", formatNumber(product.rating_count)],
    ["店铺", nullable(product.store)],
    ["分类", formatDetail(product.categories)],
  ];
  for (const [label, value] of fields) meta.append(createElement("div", {}, [createElement("span", { textContent: `${label} ` }), createElement("strong", { textContent: value })]));
  const details = createElement("div", { className: `product-details${product.metadata_missing ? " metadata-missing" : ""}` });
  details.append(createElement("p", {}, [createElement("strong", { textContent: "Features · " }), product.metadata_missing ? "未记录" : formatDetail(product.features)]));
  details.append(createElement("p", {}, [createElement("strong", { textContent: "Details · " }), product.metadata_missing ? "未记录" : formatDetail(product.details)]));
  row.append(main, meta, details);
  if (existing) {
    const reasonLabel = FEEDBACK_REASONS.find(([value]) => value === existing.reason)?.[1] || existing.reason;
    const summary = createElement("div", { className: "feedback-summary" }, [
      createElement("strong", { textContent: `已标记：${nullable(reasonLabel)}` }),
    ]);
    if (existing.note) summary.append(createElement("span", { textContent: ` · ${existing.note}` }));
    row.append(summary);
  }
  if (expanded) row.append(renderFeedbackEditor(product, turn, existing, ui));
  return row;
}

function renderFeedbackEditor(product, turn, existing, ui) {
  const key = `${turn.turn}:${product.parent_asin}`;
  const draft = ui.feedbackDrafts[key] || existing || {};
  const form = createElement("form", { className: "feedback-editor", dataset: { feedbackForm: key } });
  const fieldset = createElement("fieldset");
  fieldset.append(createElement("legend", { textContent: existing ? "更新不准确标记" : "原因" }));
  const reasonList = createElement("div", { className: "reason-list" });
  const reasonSelect = createElement("select", { name: "reason", id: `feedback-reason-${key}`, "aria-label": "原因", required: true });
  reasonSelect.append(createElement("option", { value: "", textContent: "选择原因", disabled: true, selected: !draft.reason }));
  for (const [value, label] of FEEDBACK_REASONS) {
    reasonSelect.append(createElement("option", { value, textContent: label, selected: draft.reason === value }));
  }
  reasonList.append(reasonSelect);
  fieldset.append(reasonList);
  const noteLabel = createElement("label", { htmlFor: `feedback-note-${key}`, textContent: "备注" });
  const note = createElement("textarea", { name: "note", id: `feedback-note-${key}`, rows: 2, maxlength: 2000, placeholder: "补充备注（可选）", textContent: draft.note || "", "aria-label": "备注" });
  const actions = createElement("div", { className: "feedback-actions" });
  actions.append(button("清除标记", { className: "button button-quiet", dataset: { feedbackClear: key } }));
  actions.append(button(existing ? "更新标记" : "保存标记", { className: "button button-primary", type: "submit" }));
  form.append(fieldset, noteLabel, note, actions);
  return form;
}

function renderInspector(state) {
  const content = byId("inspector-content");
  if (!content) return;
  clear(content);
  const turn = state.selectedObservation;
  if (!turn) {
    content.append(createElement("div", { className: "empty-state" }, [createElement("strong", { textContent: "暂无诊断" }), createElement("span", { textContent: "选择已完成轮次后查看 Agent 观察。" })]));
    return;
  }
  const observationState = turn.state || {};
  const trace = turn.trace || {};
  const response = turn.response || {};
  content.append(diagnosticBlock("本轮", [
    ["route", nullable(observationState.route)],
    ["intent_version", nullable(observationState.intent_version)],
    ["candidate_count", formatNumber(observationState.candidate_count)],
    ["elapsed_ms", nullable(trace.elapsed_ms ?? trace.elapsedMs ?? observationState.elapsed_ms)],
    ["ask_attribute", nullable(response.ask_attribute)],
    ["fallbacks", formatDetail(observationState.fallbacks ?? trace.fallbacks)],
  ]));
  content.append(renderConstraints(observationState.constraints));
  content.append(renderListGroup("询问与偏好", [
    ["asked_attributes", observationState.asked_attributes],
    ["pending_attribute", observationState.pending_attribute],
    ["no_preference_attributes", observationState.no_preference_attributes],
    ["query_history", observationState.query_history],
  ]));
  for (const [label, value] of [["Raw response", response], ["State", observationState], ["Trace", trace]]) {
    const details = createElement("details", { className: "raw-details" });
    details.append(createElement("summary", { textContent: label }), createElement("pre", { textContent: prettyJson(value) }));
    content.append(details);
  }
}

function diagnosticBlock(title, fields) {
  const block = createElement("section", { className: "diagnostic-group" });
  block.append(createElement("h3", { textContent: title }));
  const grid = createElement("dl", { className: "diagnostic-grid" });
  for (const [label, value] of fields) grid.append(createElement("dt", { textContent: label }), createElement("dd", { textContent: value }));
  block.append(grid);
  return block;
}

function renderConstraints(raw) {
  const block = createElement("section", { className: "diagnostic-group" });
  block.append(createElement("h3", { textContent: "Constraint ledger" }));
  const table = createElement("table", { className: "constraint-table" });
  const head = createElement("tr");
  for (const label of ["约束", "状态", "来源 / 置信度", "硬约束", "轮次 / 版本"]) head.append(createElement("th", { scope: "col", textContent: label }));
  table.append(createElement("thead", {}, [head]));
  const body = createElement("tbody");
  const entries = Array.isArray(raw) ? raw.map((item, index) => [item.key || item.name || `constraint_${index + 1}`, item]) : Object.entries(raw || {});
  if (!entries.length) body.append(createElement("tr", {}, [createElement("td", { colSpan: 5, textContent: "未记录约束" })]));
  for (const [name, item] of entries) {
    const value = item && typeof item === "object" ? item : { value: item };
    const row = createElement("tr");
    row.append(
      createElement("td", { textContent: name }),
      createElement("td", { className: `constraint-status ${value.status || ""}`, textContent: nullable(value.status, "未标记") }),
      createElement("td", { textContent: `${nullable(value.source)} / ${nullable(value.confidence)}` }),
      createElement("td", { textContent: nullable(value.hard) }),
      createElement("td", { textContent: `${nullable(value.created_turn)} / ${nullable(value.version)}` }),
    );
    body.append(row);
  }
  table.append(body);
  block.append(table);
  return block;
}

function renderListGroup(title, fields) {
  const block = createElement("section", { className: "diagnostic-group" });
  block.append(createElement("h3", { textContent: title }));
  for (const [label, value] of fields) {
    const line = createElement("div", { className: "list-value" });
    line.append(createElement("strong", { textContent: `${label}: ` }), formatDetail(value));
    block.append(line);
  }
  return block;
}

function render(state, ui) {
  const focus = captureFocus();
  const login = byId("login-view");
  const app = byId("app-view");
  if (!login || !app) return;
  const loggedIn = Boolean(state.token);
  setVisible(login, !loggedIn);
  setVisible(app, loggedIn);
  const loginError = byId("login-error");
  if (loginError) { loginError.textContent = state.loginError || ""; setVisible(loginError, Boolean(state.loginError)); }
  if (!loggedIn) { restoreFocus(focus); return; }
  const status = byId("agent-status");
  const dot = document.querySelector(".status-dot");
  if (status) status.textContent = displayStatus(state);
  if (dot) dot.className = `status-dot ${statusClass(state)}`;
  const name = byId("current-session-name");
  if (name) name.textContent = state.currentSession?.name || "未选择";
  const counter = byId("turn-counter");
  const completed = (state.turns || []).filter((turn) => turn.status === "completed").length;
  if (counter) counter.textContent = `${completed} / 10 turns`;
  const exportButton = document.querySelector('[data-action="export"]');
  const archiveButton = document.querySelector('[data-action="archive"]');
  const cloneButton = document.querySelector('[data-action="clone"]');
  if (exportButton) exportButton.disabled = !state.currentSession || state.busy;
  if (archiveButton) archiveButton.disabled = !state.currentSession || state.currentSession.archived || state.busy;
  if (cloneButton) cloneButton.disabled = !state.currentSession || state.busy;
  renderSessionControls(state);
  renderConversation(state);
  renderRecommendations(state, ui);
  renderInspector(state);
  const workspace = byId("debug-workspace");
  if (workspace) workspace.classList.toggle("inspector-open", state.inspectorOpen);
  const inspector = byId("inspector-pane");
  if (inspector) inspector.classList.toggle("is-closed", !state.inspectorOpen);
  for (const tab of MOBILE_TABS) {
    const pane = byId(`${tab === "conversation" ? "conversation" : tab}-pane`);
    if (pane) pane.classList.toggle("is-mobile-active", state.mobileTab === tab);
    const tabButton = document.querySelector(`[data-tab="${tab}"]`);
    if (tabButton) { tabButton.classList.toggle("is-active", state.mobileTab === tab); tabButton.setAttribute("aria-selected", String(state.mobileTab === tab)); }
  }
  const aria = byId("aria-status");
  if (aria) aria.textContent = state.loginError || (state.busy ? "处理中" : "");
  restoreFocus(focus);
}

function openDialog(id) { const dialog = byId(id); if (dialog?.showModal) dialog.showModal(); }
function closeDialog(id) { const dialog = byId(id); if (dialog?.open) dialog.close(); }

function prepareNewSessionDialog() {
  const name = byId("session-name");
  const tags = byId("preference-tags");
  if (name) name.value = "";
  if (tags) { tags.value = ""; tags.readOnly = false; }
}

function wire(store, ui) {
  const loginForm = byId("login-form");
  const composer = byId("message-composer");
  if (loginForm) loginForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    await store.actions.login(byId("access-token")?.value || "");
  });
  if (composer) {
    composer.addEventListener("keydown", (event) => {
      if (shouldSubmitComposerEvent(event)) { event.preventDefault(); composer.requestSubmit(); }
    });
    composer.addEventListener("submit", async (event) => {
      event.preventDefault();
      const input = byId("message-input");
      const message = input?.value || "";
      if (!message.trim()) return;
      await store.actions.send(message);
      if (input) input.value = "";
    });
  }
  document.addEventListener("click", async (event) => {
    if (!isElement(event.target)) return;
    const close = event.target.closest("[data-dialog-close]");
    if (close) { closeDialog(close.dataset.dialogClose); return; }
    const tab = event.target.closest("[data-tab]");
    if (tab) { store.actions.selectMobileTab(tab.dataset.tab); return; }
    const chip = event.target.closest("[data-session-id]");
    if (chip) { await store.actions.selectSession(chip.dataset.sessionId); return; }
    const retry = event.target.closest("[data-retry-turn]");
    if (retry) { await store.actions.retry(retry.dataset.retryTurn); return; }
    const turn = event.target.closest("[data-turn]");
    if (turn) { await store.actions.selectTurn(turn.dataset.turn); return; }
    const toggle = event.target.closest("[data-feedback-toggle]");
    if (toggle) { ui.feedbackKey = ui.feedbackKey === toggle.dataset.feedbackToggle ? null : toggle.dataset.feedbackToggle; render(store.getState(), ui); return; }
    const clearFeedback = event.target.closest("[data-feedback-clear]");
    if (clearFeedback) {
      const [turnNumber, ...asinParts] = clearFeedback.dataset.feedbackClear.split(":");
      await store.actions.feedback(Number(turnNumber), asinParts.join(":"), { incorrect: false, note: "" });
      delete ui.feedbackDrafts[clearFeedback.dataset.feedbackClear];
      ui.feedbackKey = null;
      render(store.getState(), ui);
      return;
    }
    const action = event.target.closest("[data-action]");
    if (!action) return;
    switch (action.dataset.action) {
      case "new-session": prepareNewSessionDialog(); openDialog("new-session-dialog"); break;
      case "import": openDialog("import-dialog"); break;
      case "export": {
        const payload = await store.actions.export();
        if (payload) { const result = exportBlob(payload); downloadBlob(result.blob, result.filename); }
        break;
      }
      case "archive": openDialog("archive-dialog"); break;
      case "clone": openDialog("clone-dialog"); break;
      case "logout": store.actions.logout(); break;
      case "toggle-inspector": store.actions.toggleInspector(); break;
      default: break;
    }
  });
  document.addEventListener("change", async (event) => {
    if (!isElement(event.target)) return;
    if (event.target.id === "session-select") await store.actions.selectSession(event.target.value);
  });
  document.addEventListener("submit", async (event) => {
    if (!isElement(event.target)) return;
    const form = event.target;
    if (form.id === "new-session-form") {
      event.preventDefault();
      const name = byId("session-name")?.value || "Untitled session";
      const tags = normalizeTags(byId("preference-tags")?.value || "");
      closeDialog("new-session-dialog");
      await store.actions.create({ name, preferenceTags: tags });
    } else if (form.id === "import-form") {
      event.preventDefault();
      const errorNode = byId("import-error");
      try {
        const file = byId("import-file")?.files?.[0];
        const payload = await readImportFile(file);
        closeDialog("import-dialog");
        await store.actions.import(payload);
      } catch (error) {
        if (errorNode) { errorNode.textContent = error.message || "导入失败"; setVisible(errorNode, true); }
      }
    } else if (form.id === "archive-form") {
      event.preventDefault(); closeDialog("archive-dialog"); await store.actions.archive(true);
    } else if (form.id === "clone-form") {
      event.preventDefault();
      const raw = byId("clone-through-turn")?.value;
      closeDialog("clone-dialog");
      await store.actions.clone(raw === "" ? undefined : Number(raw));
    } else if (form.dataset.feedbackForm) {
      event.preventDefault();
      const [turnNumber, ...asinParts] = form.dataset.feedbackForm.split(":");
      const reason = form.querySelector("select[name=reason]")?.value;
      const note = form.querySelector("textarea")?.value || "";
      ui.feedbackDrafts[form.dataset.feedbackForm] = { reason, note };
      await store.actions.feedback(Number(turnNumber), asinParts.join(":"), { incorrect: true, reason, note });
      ui.feedbackKey = null;
      render(store.getState(), ui);
    }
  });
}

export function startApp({ service = null } = {}) {
  if (typeof document === "undefined") return null;
  const api = service || createApi();
  const store = createStore({ service: api, initialState: { token: api.getToken?.() || null } });
  const ui = { feedbackKey: null, feedbackDrafts: Object.create(null) };
  store.subscribe((state) => render(state, ui));
  wire(store, ui);
  render(store.getState(), ui);
  if (store.getState().token) store.actions.bootstrap();
  return { store, api };
}

if (typeof document !== "undefined") {
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", () => startApp(), { once: true });
  else startApp();
}
