import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { createStore } from "../../src/compasscart_debug/static/js/store.js";
import { shouldSubmitComposerEvent } from "../../src/compasscart_debug/static/js/app.js";

const session = {
  session_id: "session-1",
  name: "Weekend kit",
  profile: { preference_tags: ["lightweight"] },
  archived: false,
  dirty: false,
  read_only_reason: null,
};

function turn(number, message, asin) {
  return {
    session_id: session.session_id,
    turn: number,
    request_id: `request-${number}`,
    status: "completed",
    user_message: message,
    response: {
      message: `Agent answer ${number}`,
      ask_attribute: null,
      recommendations: [{ parent_asin: asin }],
      usage: { total_tokens: number },
    },
    products: [{
      rank: 1,
      parent_asin: asin,
      title: `Product ${number}`,
      price: 42,
      rating: 4.5,
      rating_count: 10,
      store: "Demo store",
      categories: ["Demo"],
      features: ["Useful"],
      details: {},
      metadata_missing: false,
    }],
    state: {
      turn: number,
      route: "recommend",
      intent_version: 1,
      constraints: {},
      asked_attributes: [],
      pending_attribute: null,
      no_preference_attributes: [],
      query_history: [message],
      candidate_count: 1,
    },
    trace: { route: "recommend", elapsed_ms: number * 10 },
    feedback: [],
    error: null,
  };
}

test("selecting a turn updates observation, recommendations, and diagnostics atomically", async () => {
  const first = turn(1, "first", "ASIN-1");
  const second = turn(2, "second", "ASIN-2");
  const service = {
    detail: async () => ({ session, turns: [first, second], continuation: "ready", can_send: true }),
  };
  const store = createStore({ service, initialState: { token: "token" } });

  await store.actions.selectSession(session.session_id);
  await store.actions.selectTurn(2);
  const state = store.getState();

  assert.equal(state.selectedTurn, 2);
  assert.equal(state.selectedObservation, second);
  assert.equal(state.observation, second);
  assert.deepEqual(state.recommendations, second.products);
  assert.deepEqual(state.diagnostics, second.state);
});

test("store accepts a compact initial session and direct selectTurn action", () => {
  const store = createStore({ currentSession: { turns: [{ turn: 1 }, { turn: 2 }] } });
  store.selectTurn(2);
  assert.equal(store.getState().selectedTurn, 2);
  assert.equal(store.getState().selectedObservation.turn, 2);
});

test("an authentication error clears only the token", async () => {
  const service = {
    health: async () => { throw Object.assign(new Error("unauthorized"), { status: 401 }); },
  };
  const store = createStore({
    service,
    initialState: {
      token: "stale",
      currentSession: session,
      sessions: [session],
      selectedTurn: 4,
      loginError: null,
    },
  });

  await store.actions.login("bad-token");
  const state = store.getState();
  assert.equal(state.token, null);
  assert.deepEqual(state.currentSession, session);
  assert.deepEqual(state.sessions, [session]);
  assert.equal(state.selectedTurn, 4);
});

test("composer submits Enter but preserves Shift+Enter and IME composition", () => {
  assert.equal(shouldSubmitComposerEvent({ key: "Enter", shiftKey: false, isComposing: false }), true);
  assert.equal(shouldSubmitComposerEvent({ key: "Enter", shiftKey: true, isComposing: false }), false);
  assert.equal(shouldSubmitComposerEvent({ key: "Enter", shiftKey: false, isComposing: true }), false);
  assert.equal(shouldSubmitComposerEvent({ key: "a", shiftKey: false, isComposing: false }), false);
});

test("client JavaScript never uses unsafe HTML mutation or eval", async () => {
  const root = resolve("src/compasscart_debug/static/js");
  const files = ["api.js", "dom.js", "store.js", "app.js"];
  const source = (await Promise.all(files.map((file) => readFile(resolve(root, file), "utf8")))).join("\n");
  assert.doesNotMatch(source, /\b(?:innerHTML|outerHTML|insertAdjacentHTML)\b/);
  assert.doesNotMatch(source, /\beval\s*\(/);
});
