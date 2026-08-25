import { ApiError } from "./api.js";

const DEFAULT_STATE = {
  token: null,
  service: null,
  sessions: [],
  currentSession: null,
  sessionDetail: null,
  turns: [],
  selectedTurn: null,
  selectedObservation: null,
  observation: null,
  recommendations: [],
  diagnostics: null,
  mobileTab: "conversation",
  inspectorOpen: false,
  busy: false,
  loginError: null,
  serviceStatus: "unknown",
  continuation: null,
  canSend: false,
};

function authError(error) {
  return Number(error?.status) === 401 || error?.code === "authentication_failed" || error?.code === "unauthorized";
}

function errorMessage(error) {
  if (error instanceof ApiError || error?.message) return String(error.message);
  return "The debug service could not complete that action.";
}

function copyState(state, patch) {
  return { ...state, ...patch };
}

function turnsFrom(detail) {
  return Array.isArray(detail?.turns) ? detail.turns.slice() : [];
}

function latestTurn(turns) {
  return turns.length ? turns[turns.length - 1] : null;
}

function applyTurn(state, turn) {
  const observation = turn || null;
  return {
    selectedTurn: observation?.turn ?? null,
    selectedObservation: observation,
    observation,
    recommendations: Array.isArray(observation?.products) ? observation.products : [],
    diagnostics: observation?.state || null,
  };
}

function mergeSession(sessions, session) {
  if (!session) return sessions;
  const index = sessions.findIndex((item) => item.session_id === session.session_id);
  if (index < 0) return [...sessions, session];
  const next = sessions.slice();
  next[index] = session;
  return next;
}

export function createStore(options = {}) {
  const service = options.service || null;
  const initialState = options.initialState || Object.fromEntries(
    Object.entries(options).filter(([key]) => key !== "service"),
  );
  const suppliedTurns = Array.isArray(initialState.turns)
    ? initialState.turns
    : (Array.isArray(initialState.currentSession?.turns) ? initialState.currentSession.turns : []);
  let state = {
    ...DEFAULT_STATE,
    ...initialState,
    turns: suppliedTurns.slice(),
    service: service || initialState.service || null,
  };
  const listeners = new Set();

  function emit() {
    for (const listener of listeners) listener(state);
    return state;
  }

  function set(patch) {
    state = copyState(state, patch);
    return emit();
  }

  function setError(error, { preserve = true } = {}) {
    if (authError(error)) {
      state.service?.clearToken?.();
      state = copyState(state, { token: null, loginError: errorMessage(error), busy: false });
      return emit();
    }
    const patch = { loginError: errorMessage(error), busy: false };
    if (!preserve) Object.assign(patch, { currentSession: null, sessionDetail: null, turns: [], ...applyTurn(state, null) });
    return set(patch);
  }

  async function loadDetail(sessionId, { selectLatest = true } = {}) {
    const detail = await state.service.detail(sessionId);
    const turns = turnsFrom(detail);
    const currentSession = detail?.session || state.sessions.find((item) => item.session_id === sessionId) || null;
    const selected = selectLatest ? latestTurn(turns) : (state.selectedTurn ? turns.find((item) => item.turn === state.selectedTurn) : latestTurn(turns));
    state = {
      ...state,
      currentSession,
      sessionDetail: detail,
      turns,
      continuation: detail?.continuation || null,
      canSend: Boolean(detail?.can_send),
      loginError: null,
      ...applyTurn(state, selected),
    };
    return emit();
  }

  const actions = {
    async bootstrap() {
      if (!state.service) return state;
      try {
        set({ busy: true, serviceStatus: "checking", loginError: null });
        const health = await state.service.health("ready");
        const listed = await state.service.sessions("active");
        const sessions = Array.isArray(listed?.sessions) ? listed.sessions : [];
        state = copyState(state, { token: state.service.getToken?.() || state.token, sessions, serviceStatus: health?.status || "ready", busy: false });
        emit();
        if (sessions.length && !state.currentSession) await loadDetail(sessions[0].session_id);
        return state;
      } catch (error) {
        setError(error);
        return state;
      }
    },

    async login(token) {
      if (!state.service) return state;
      const clean = typeof token === "string" ? token.trim() : "";
      state.service.setToken?.(clean);
      set({ busy: true, loginError: null, token: clean || null });
      try {
        const health = await state.service.health("ready");
        const listed = await state.service.sessions("active");
        const sessions = Array.isArray(listed?.sessions) ? listed.sessions : [];
        state = copyState(state, { token: state.service.getToken?.() || clean || null, sessions, serviceStatus: health?.status || "ready", busy: false, loginError: null });
        emit();
        if (sessions.length) await loadDetail(sessions[0].session_id);
        return state;
      } catch (error) {
        setError(error);
        return state;
      }
    },

    logout() {
      state.service?.clearToken?.();
      return set({ token: null, loginError: null, busy: false });
    },

    async selectSession(sessionId) {
      if (!state.service || !sessionId) return state;
      try {
        set({ busy: true, loginError: null });
        await loadDetail(sessionId);
        return set({ busy: false });
      } catch (error) {
        setError(error);
        return state;
      }
    },

    async selectTurn(turnNumber) {
      const selected = state.turns.find((turn) => Number(turn.turn) === Number(turnNumber)) || null;
      return set({ ...applyTurn(state, selected), loginError: null });
    },

    selectMobileTab(tab) {
      return set({ mobileTab: ["conversation", "recommendations", "inspector"].includes(tab) ? tab : "conversation" });
    },

    toggleInspector(open = !state.inspectorOpen) {
      return set({ inspectorOpen: Boolean(open) });
    },

    async send(message, { requestId = null } = {}) {
      if (!state.service || !state.currentSession || !String(message || "").trim()) return state;
      const id = requestId || (globalThis.crypto?.randomUUID ? globalThis.crypto.randomUUID() : `debug-${Date.now()}-${Math.random().toString(16).slice(2)}`);
      const sessionId = state.currentSession.session_id;
      try {
        set({ busy: true, loginError: null });
        await state.service.send(sessionId, id, message);
        await loadDetail(sessionId);
        return set({ busy: false });
      } catch (error) {
        setError(error);
        try { await loadDetail(sessionId, { selectLatest: true }); } catch { /* retain the server error in state */ }
        return state;
      }
    },

    async retry(turnOrNumber) {
      const turn = typeof turnOrNumber === "object" ? turnOrNumber : state.turns.find((item) => Number(item.turn) === Number(turnOrNumber));
      if (!turn) return state;
      return actions.send(turn.user_message, { requestId: turn.request_id });
    },

    async create({ name, preferenceTags = [] } = {}) {
      if (!state.service) return state;
      try {
        set({ busy: true, loginError: null });
        const payload = await state.service.create(name || "Untitled session", { preference_tags: preferenceTags });
        const created = payload?.session || null;
        state = copyState(state, { sessions: mergeSession(state.sessions, created), busy: false });
        emit();
        if (created) await loadDetail(created.session_id);
        return state;
      } catch (error) {
        setError(error);
        return state;
      }
    },

    async feedback(turn, parentAsin, values) {
      if (!state.service || !state.currentSession) return state;
      try {
        set({ busy: true, loginError: null });
        await state.service.feedback(state.currentSession.session_id, turn, parentAsin, values);
        await loadDetail(state.currentSession.session_id, { selectLatest: false });
        return set({ busy: false });
      } catch (error) {
        setError(error);
        return state;
      }
    },

    async import(payload) {
      if (!state.service) return state;
      try {
        set({ busy: true, loginError: null });
        const detail = await state.service.importSession(payload);
        const imported = detail?.session || null;
        state = copyState(state, { sessions: mergeSession(state.sessions, imported), busy: false });
        emit();
        if (imported) await loadDetail(imported.session_id);
        return state;
      } catch (error) {
        setError(error);
        return state;
      }
    },

    async export() {
      if (!state.service || !state.currentSession) return null;
      try {
        set({ busy: true, loginError: null });
        const payload = await state.service.exportSession(state.currentSession.session_id);
        set({ busy: false });
        return payload;
      } catch (error) {
        setError(error);
        return null;
      }
    },

    async archive(archived = true) {
      if (!state.service || !state.currentSession) return state;
      try {
        set({ busy: true, loginError: null });
        const payload = await state.service.patch(state.currentSession.session_id, { archived });
        const updated = payload?.session || state.currentSession;
        state = copyState(state, { currentSession: updated, sessions: mergeSession(state.sessions, updated), busy: false });
        emit();
        return state;
      } catch (error) {
        setError(error);
        return state;
      }
    },

    async clone(throughTurn = undefined) {
      if (!state.service || !state.currentSession) return state;
      try {
        set({ busy: true, loginError: null });
        const detail = await state.service.clone(state.currentSession.session_id, throughTurn);
        const cloned = detail?.session || null;
        state = copyState(state, { sessions: mergeSession(state.sessions, cloned), busy: false });
        emit();
        if (cloned) await loadDetail(cloned.session_id);
        return state;
      } catch (error) {
        setError(error);
        return state;
      }
    },
  };

  const store = {
    getState: () => state,
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    actions,
  };
  // Keep the compact direct-action spelling useful for small consumers and
  // Node tests while the UI uses the namespaced actions object.
  for (const [name, action] of Object.entries(actions)) store[name] = action;
  Object.defineProperty(store, "state", { enumerable: true, get: () => state });
  return store;
}
