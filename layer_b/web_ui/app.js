/* Shared front-end helpers for every PicarX console page (served at /app.js).
 * Exposes a small global `PX`: fetch helpers, the top-nav + header, a single
 * shared /state poll loop pages subscribe to, and the conversation-log
 * renderer (with the ✓/✗ command-feedback and object-relabel affordances)
 * used by both the Dashboard and Training pages. */
const PX = (() => {
  const NAV = [
    ["/", "Dashboard"], ["/drive", "Drive & Cam"], ["/training", "Training"],
    ["/people", "People"], ["/audio", "Audio"], ["/config", "Config"],
  ];
  const pollers = [];
  let logEntries = [];

  async function post(path, body) {
    try {
      const r = await fetch(path, {method: "POST",
        headers: {"Content-Type": "application/json"}, body: JSON.stringify(body || {})});
      // Return the parsed body on any HTTP status so callers can read .ok
      // (200) or .error (4xx/5xx); null means the request never landed.
      return await r.json().catch(() => ({ok: r.ok}));
    } catch (e) { return null; }
  }
  async function get(path) {
    const r = await fetch(path);
    if (!r.ok) throw new Error(r.status);
    return r.json();
  }
  function esc(s) { const d = document.createElement("span"); d.textContent = s == null ? "" : s; return d.innerHTML; }
  function say(text) { return post("/say", {text}); }

  // Build the sticky header (brand + connection + battery + nav) at the top of
  // <body> so every page shares one chrome without repeating the markup.
  function mountHeader() {
    const here = location.pathname === "/index.html" ? "/" : location.pathname;
    const nav = NAV.map(([href, label]) =>
      `<a href="${href}" class="${href === here ? "active" : ""}">${label}</a>`).join("");
    const h = document.createElement("header");
    h.innerHTML =
      `<div class="brand"><span>PicarX</span>` +
      `<span><span id="h-batt">–</span> · <span id="conn">connecting…</span></span></div>` +
      `<nav>${nav}</nav>`;
    document.body.prepend(h);
  }

  function onPoll(cb) { pollers.push(cb); }

  function applyHeader(s) {
    const conn = document.getElementById("conn");
    conn.textContent = "connected"; conn.className = "ok";
    const batt = document.getElementById("h-batt");
    if (s.battery_v != null) {
      batt.textContent = s.battery_v.toFixed(2) + " V";
      batt.className = s.battery_low ? "low" : "";
    } else { batt.textContent = "–"; batt.className = ""; }
  }
  async function pollOnce() {
    try {
      const s = await get("/state");
      applyHeader(s);
      pollers.forEach(cb => { try { cb(s); } catch (e) {} });
    } catch (e) {
      const conn = document.getElementById("conn");
      if (conn) { conn.textContent = "disconnected"; conn.className = "bad"; }
    }
  }
  // Any page can trigger an immediate refresh (e.g. right after a POST).
  function poll() { return pollOnce(); }
  function startPolling(ms) { pollOnce(); setInterval(pollOnce, ms || 2000); }

  // ---- shared conversation log + feedback (Dashboard, Training) ----
  function renderLog(elId, entries) {
    logEntries = entries || [];
    const el = document.getElementById(elId);
    if (!el) return;
    el.innerHTML = logEntries.map((e, i) => {
      let fb = "";
      if (e.kind === "robot") {
        if (e.fb) {
          fb = `<span class="fb ${e.fb}">${e.fb === "correct" ? "✓" : "✗"}</span>`;
        } else if (e.obs && ((e.obs.items && e.obs.items.length) || e.obs.subject)) {
          fb = `<button class="fbbtn ok" data-obs="yes" data-idx="${i}" title="right - that's what it is">✓</button>` +
               `<button class="fbbtn bad" data-obs="no" data-idx="${i}" title="wrong - tell me what it is">✗</button>`;
        } else {
          fb = `<button class="fbbtn ok" data-fb="correct" data-idx="${i}" title="understood me">✓</button>` +
               `<button class="fbbtn bad" data-fb="incorrect" data-idx="${i}" title="misunderstood me">✗</button>`;
        }
      }
      return `<div class="${e.kind}">${e.t} ${e.kind === "robot" ? "🤖" : "🗣"} ${esc(e.text)}${fb}</div>`;
    }).join("");
  }
  function bindLog(elId) {
    const el = document.getElementById(elId);
    if (!el) return;
    el.addEventListener("click", async ev => {
      const obsBtn = ev.target.closest("button[data-obs]");
      if (obsBtn) {
        const e = logEntries[+obsBtn.dataset.idx];
        const items = (e.obs && e.obs.items) || [];
        let item = items[0] || {label: (e.obs && e.obs.subject) || "", id: null};
        if (obsBtn.dataset.obs === "no" && items.length > 1) {
          const which = (prompt("Which did I get wrong? " +
            items.map(o => o.label).join(", ")) || "").trim().toLowerCase();
          item = items.find(o => (o.label || "").toLowerCase() === which) || item;
        }
        const guess = item.label || "";
        let label = guess;
        if (obsBtn.dataset.obs === "no") {
          label = (prompt("What is it actually?") || "").trim();
          if (!label) return;
        }
        await post("/label", {label, guess, object_id: item.id, response: e.text});
        poll(); return;
      }
      const btn = ev.target.closest("button[data-fb]");
      if (!btn) return;
      const e = logEntries[+btn.dataset.idx];
      const verdict = btn.dataset.fb;
      let correction = "";
      if (verdict === "incorrect") {
        correction = (prompt("What did you want the robot to do?\n" +
          "(it will do it now AND learn the phrasing - leave blank to just flag it)") || "").trim();
      }
      await post("/feedback", {verdict, utterance: e.re || "", response: e.text, correction});
      poll();
    });
  }

  return {post, get, esc, say, mountHeader, onPoll, poll, startPolling,
          renderLog, bindLog};
})();

document.addEventListener("DOMContentLoaded", () => PX.mountHeader());
