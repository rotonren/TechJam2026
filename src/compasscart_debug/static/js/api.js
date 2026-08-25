const TOKEN_KEY = "compasscart_debug_token";

export class ApiError extends Error {
  constructor(message, { status = 0, code = "request_failed", retryable = false, fieldErrors = null } = {}) {
    super(message || "Request failed.");
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.retryable = Boolean(retryable);
    this.fieldErrors = fieldErrors && typeof fieldErrors === "object" ? fieldErrors : null;
  }
}

function storageOrDefault(storage) {
  if (storage) return storage;
  try {
    return window.sessionStorage;
  } catch {
    return null;
  }
}

function safeToken(storage) {
  try {
    return storage?.getItem(TOKEN_KEY) || null;
  } catch {
    return null;
  }
}

function setStoredToken(storage, token) {
  try {
    if (token) storage?.setItem(TOKEN_KEY, token);
    else storage?.removeItem(TOKEN_KEY);
  } catch {
    // A blocked sessionStorage should not prevent a local demo from opening.
  }
}

function safePath(value) {
  return encodeURIComponent(String(value));
}

async function parsePayload(response) {
  const contentType = response.headers?.get?.("content-type") || "";
  if (contentType.includes("json")) {
    try {
      return await response.json();
    } catch {
      return null;
    }
  }
  try {
    const body = await response.text();
    return body ? { message: body } : null;
  } catch {
    return null;
  }
}

function errorFromResponse(response, payload) {
  const envelope = payload && typeof payload === "object" ? payload.error : null;
  const source = envelope && typeof envelope === "object" ? envelope : payload;
  const message = source && typeof source.message === "string" ? source.message : "Request failed.";
  const code = source && typeof source.code === "string" ? source.code : "request_failed";
  const retryable = Boolean(source && source.retryable);
  const fieldErrors = source && typeof source.field_errors === "object" ? source.field_errors : null;
  return new ApiError(message, { status: response.status, code, retryable, fieldErrors });
}

export function createApi({ fetchImpl = globalThis.fetch, storage = null, base = "" } = {}) {
  const tokenStorage = storageOrDefault(storage);
  let token = safeToken(tokenStorage);

  async function request(path, { method = "GET", body, signal } = {}) {
    if (typeof fetchImpl !== "function") throw new ApiError("Fetch is unavailable.");
    const headers = { Accept: "application/json" };
    if (token) headers.Authorization = `Bearer ${token}`;
    const init = { method, headers, signal };
    if (body !== undefined) {
      headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(body);
    }
    let response;
    try {
      response = await fetchImpl(`${base}${path}`, init);
    } catch {
      throw new ApiError("Unable to reach the debug service.", { retryable: true });
    }
    const payload = await parsePayload(response);
    if (!response.ok) throw errorFromResponse(response, payload);
    return payload ?? {};
  }

  const api = {
    tokenKey: TOKEN_KEY,
    getToken() { return token || safeToken(tokenStorage); },
    setToken(value) {
      token = typeof value === "string" && value.trim() ? value.trim() : null;
      setStoredToken(tokenStorage, token);
      return token;
    },
    clearToken() {
      token = null;
      setStoredToken(tokenStorage, null);
    },
    request,
    async health(kind = "ready") { return request(`/api/health/${kind}`); },
    async sessions(scope = "active") { return request(`/api/sessions?scope=${encodeURIComponent(scope)}`); },
    async detail(sessionId) { return request(`/api/sessions/${safePath(sessionId)}`); },
    async create(name, profile) { return request("/api/sessions", { method: "POST", body: { name, profile } }); },
    async patch(sessionId, values) { return request(`/api/sessions/${safePath(sessionId)}`, { method: "PATCH", body: values }); },
    async send(sessionId, requestId, userMessage) {
      return request(`/api/sessions/${safePath(sessionId)}/messages`, {
        method: "POST",
        body: { request_id: requestId, user_message: userMessage },
      });
    },
    async feedback(sessionId, turn, parentAsin, values) {
      return request(`/api/sessions/${safePath(sessionId)}/turns/${encodeURIComponent(turn)}/feedback/${safePath(parentAsin)}`, {
        method: "PUT",
        body: { incorrect: Boolean(values?.incorrect ?? values?.is_inaccurate), reason: values?.reason, note: values?.note || "" },
      });
    },
    async exportSession(sessionId) { return request(`/api/sessions/${safePath(sessionId)}/export`); },
    async importSession(payload) {
      const normalized = payload && typeof payload.text === "function" ? await readImportFile(payload) : payload;
      return request("/api/import", { method: "POST", body: normalized });
    },
    async clone(sessionId, throughTurn = undefined) {
      const body = throughTurn === undefined || throughTurn === null ? undefined : { through_turn: throughTurn };
      return request(`/api/sessions/${safePath(sessionId)}/clone`, { method: "POST", body });
    },
  };
  api.export = api.exportSession;
  api.import = api.importSession;
  return api;
}

export async function readImportFile(file) {
  if (!file || typeof file.text !== "function") throw new ApiError("Choose a JSON export file.", { code: "invalid_file" });
  let raw;
  try {
    raw = await file.text();
    return JSON.parse(raw);
  } catch {
    throw new ApiError("The export file is not valid JSON.", { code: "invalid_file" });
  }
}

export function exportBlob(payload, filename = "compasscart-debug-session.json") {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  return { blob, filename };
}

export function downloadBlob(blob, filename = "compasscart-debug-session.json", documentRef = globalThis.document) {
  if (!documentRef || !documentRef.createElement) return false;
  const url = URL.createObjectURL(blob);
  const link = documentRef.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
  return true;
}

export const api = createApi();
export { TOKEN_KEY };
