const fs = require("fs");
const path = require("path");
const { writeBuild, BUILD, SIBLING_TOOLS } = require("./lib/io");

const USAGE_LOG_CANDIDATES = [
  path.join(SIBLING_TOOLS, "shortcut_usage.jsonl"),
  path.join(SIBLING_TOOLS, "runtime", "shortcut_usage.jsonl"),
];
const EVENTS_LOG_CANDIDATES = [
  path.join(SIBLING_TOOLS, "charybdis_events.jsonl"),
  path.join(SIBLING_TOOLS, "runtime", "charybdis_events.jsonl"),
];
const USAGE_LOG = USAGE_LOG_CANDIDATES.find((p) => fs.existsSync(p)) || USAGE_LOG_CANDIDATES[0];
const EVENTS_LOG = EVENTS_LOG_CANDIDATES.find((p) => fs.existsSync(p)) || EVENTS_LOG_CANDIDATES[0];
const SCORES_PATH = path.join(BUILD, "app_shortcut_scores.json");

/**
 * Normalize AHK key notation to human-readable format matching app_shortcut_scores.json.
 * AHK: ^l, ^+p, #{Space}, !{F4}  =>  Ctrl+L, Ctrl+Shift+P, Win+Space, Alt+F4
 */
function normalizeKeys(ahk) {
  let mods = [];
  let i = 0;
  while (i < ahk.length) {
    if (ahk[i] === "^") { mods.push("Ctrl"); i++; }
    else if (ahk[i] === "!") { mods.push("Alt"); i++; }
    else if (ahk[i] === "#") { mods.push("Win"); i++; }
    else if (ahk[i] === "+") { mods.push("Shift"); i++; }
    else break;
  }
  let rest = ahk.slice(i);
  // Strip AHK braces: {Space} => Space, {F4} => F4, {Enter} => Enter
  const braceMatch = rest.match(/^\{(.+)\}$/);
  if (braceMatch) rest = braceMatch[1];
  // Capitalize single letters, title-case known keys
  if (rest.length === 1) rest = rest.toUpperCase();
  else if (/^f\d+$/i.test(rest)) rest = rest.toUpperCase();
  else rest = rest.charAt(0).toUpperCase() + rest.slice(1);
  // Special AHK names to standard
  const nameMap = { "``": "`", "Escape": "Esc", "Delete": "Del", "Backspace": "Backspace" };
  if (nameMap[rest]) rest = nameMap[rest];
  return [...mods, rest].join("+");
}

function run(config) {
  const errors = [], warnings = [];

  if (!fs.existsSync(USAGE_LOG)) {
    const empty = {
      timestamp: new Date().toISOString(),
      total_events: 0, period_days: 0,
      shortcuts: {}, sequences: {}, shortcut_sequences: {}, chains: {}, workflows: {},
      shortcut_workflows: {}, app_sequences: {}, app_workflows: {}, by_app: {}, by_layer: {},
      by_layer_shortcut: {}, layer_shortcuts: {},
      raw_completion_keys: {}, raw_completion_total: 0,
      note: "No usage data yet. Start charybdis_helpers.ahk to begin collecting.",
    };
    writeBuild("usage_stats.json", empty);
    return { success: true, output: { summary: "No usage data (shortcut_usage.jsonl not found)" }, errors, warnings };
  }

  let raw = fs.readFileSync(USAGE_LOG, "utf-8");
  if (raw.charCodeAt(0) === 0xFEFF) raw = raw.slice(1);
  const lines = raw.split(/\r?\n/).filter(Boolean);
  const shortcuts = {};
  const sequences = {};
  const byApp = {};
  const byLayer = {};
  const mouseClicks = {};
  const mouseByLayer = {};
  const scrollEvents = {};
  const scrollByLayer = {};
  let scrollTotal = 0;
  const layerSessions = {};
  const holdHeavy = {};
  const modifierTaps = {};
  const appFocusTime = {};
  const chainBuffer = [];  // ring buffer for 3-event chain detection
  const chains = {};
  const shortcutSequences = {};
  const shortcutWorkflows = {};
  const appSequences = {};
  const appWorkflows = {};
  const rawCompletionKeys = {};
  let rawCompletionTotal = 0;
  const shortcutEventLog = [];  // flat log for workflow detection
  let earliest = null, latest = null;

  // New optimizer v2 data structures
  const mouseSessions = [];
  const corrections = [];
  const modifierErrors = [];
  const layerTransitions = [];
  const layerBounces = [];
  const layerSticky = [];
  const shortcutRetries = {};
  const shortcutNoopHints = {};
  const byContext = {};
  const byHand = {};
  const shortcutFirstSeen = {};
  const shortcutConfidence = {};  // gap_ms distribution per sequence
  const sequenceGaps = {};  // raw gap_ms per sequence for CV computation

  function maybeNormalizeKeys(value) {
    if (!value) return "";
    const s = String(value);
    return /[\^\!\#\{]/.test(s) ? normalizeKeys(s) : s;
  }

  function addMapCount(obj, key, inc = 1) {
    if (!key) return;
    obj[key] = (obj[key] || 0) + inc;
  }

  function rawCompletionBaseName(keys, explicitBase) {
    const explicit = String(explicitBase || "").trim();
    const valid = new Set([
      "Dash and Underscore",
      "Equals and Plus",
      "Grave Accent and Tilde",
      "Right Brace",
      "PageUp",
      "PageDown",
      "Home",
      "End",
    ]);
    if (valid.has(explicit)) return explicit;
    const norm = maybeNormalizeKeys(keys || "");
    const base = norm.split("+").pop();
    const map = {
      "-": "Dash and Underscore",
      "=": "Equals and Plus",
      "`": "Grave Accent and Tilde",
      "]": "Right Brace",
      "PgUp": "PageUp",
      "PageUp": "PageUp",
      "Page Up": "PageUp",
      "PgDn": "PageDown",
      "PageDn": "PageDown",
      "PageDown": "PageDown",
      "Page Down": "PageDown",
      "Home": "Home",
      "End": "End",
    };
    return map[base] || "";
  }

  function addRawCompletionUsage(keys, baseKey, count, app, layer) {
    const base = rawCompletionBaseName(keys, baseKey);
    if (!base) return;
    const cnt = Number(count) || 1;
    if (!rawCompletionKeys[base]) {
      rawCompletionKeys[base] = { count: 0, apps: {}, by_layer: {}, variants: {} };
    }
    const row = rawCompletionKeys[base];
    row.count += cnt;
    rawCompletionTotal += cnt;
    addMapCount(row.apps, app, cnt);
    addMapCount(row.by_layer, String(layer || "0"), cnt);
    addMapCount(row.variants, maybeNormalizeKeys(keys || base), cnt);
  }

  function percentile(values, p) {
    if (!values.length) return 0;
    const sorted = values.slice().sort((a, b) => a - b);
    const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor((sorted.length - 1) * p)));
    return sorted[idx];
  }

  function sortedAppsKey(apps) {
    return [...new Set((apps || []).filter(Boolean).map(a => String(a).toLowerCase()))].sort().join(" + ");
  }

  function addShortcutSequence(from, to, gapMs, fromApp, toApp, sameApp) {
    const keyA = maybeNormalizeKeys(from);
    const keyB = maybeNormalizeKeys(to);
    if (!keyA || !keyB) return;
    const seqKey = `${keyA} -> ${keyB}`;
    if (!shortcutSequences[seqKey]) {
      shortcutSequences[seqKey] = {
        count: 0,
        gaps: [],
        same_app_count: 0,
        cross_app_count: 0,
        apps: {},
      };
    }
    const row = shortcutSequences[seqKey];
    row.count++;
    if (gapMs || gapMs === 0) row.gaps.push(Number(gapMs) || 0);
    const isSameApp = sameApp !== undefined ? !!sameApp : fromApp && toApp && fromApp === toApp;
    if (isSameApp) row.same_app_count++;
    else row.cross_app_count++;
    addMapCount(row.apps, fromApp);
    addMapCount(row.apps, toApp);

    if (!sequences[seqKey]) sequences[seqKey] = { count: 0, total_gap_ms: 0 };
    sequences[seqKey].count++;
    sequences[seqKey].total_gap_ms += Number(gapMs) || 0;
    if (!sequenceGaps[seqKey]) sequenceGaps[seqKey] = [];
    if (gapMs || gapMs === 0) sequenceGaps[seqKey].push(Number(gapMs) || 0);
  }

  function addShortcutWorkflow(keys, apps, layers, spanMs) {
    const normKeys = (keys || []).map(maybeNormalizeKeys).filter(Boolean);
    if (normKeys.length < 3) return;
    const wfKey = normKeys.join(" -> ");
    if (!shortcutWorkflows[wfKey]) {
      shortcutWorkflows[wfKey] = {
        count: 0,
        total_span_ms: 0,
        apps: {},
        app_count: 0,
        layer_count: 0,
      };
    }
    const row = shortcutWorkflows[wfKey];
    row.count++;
    row.total_span_ms += Number(spanMs) || 0;
    for (const appName of apps || []) addMapCount(row.apps, appName);
    row.app_count = Math.max(row.app_count, new Set((apps || []).filter(Boolean)).size);
    row.layer_count = Math.max(row.layer_count, new Set((layers || []).map(String)).size);

    const uniqueApps = new Set((apps || []).filter(Boolean).map(a => String(a).toLowerCase()));
    if (uniqueApps.size >= 2) {
      let switchCount = 0;
      for (let i = 0; i < (apps || []).length - 1; i++) {
        if (String(apps[i]).toLowerCase() !== String(apps[i + 1]).toLowerCase()) switchCount++;
      }
      addAppWorkflow(apps, switchCount, normKeys.length, spanMs);
    }

    if (!chains[wfKey]) chains[wfKey] = { count: 0, total_ms: 0 };
    chains[wfKey].count++;
    chains[wfKey].total_ms += Number(spanMs) || 0;
  }

  function addAppSequence(fromApp, toApp, prevDurationMs) {
    if (!fromApp || !toApp) return;
    const seqKey = `${String(fromApp).toLowerCase()} -> ${String(toApp).toLowerCase()}`;
    if (!appSequences[seqKey]) appSequences[seqKey] = { count: 0, total_prev_duration_ms: 0 };
    appSequences[seqKey].count++;
    appSequences[seqKey].total_prev_duration_ms += Number(prevDurationMs) || 0;
  }

  function addAppWorkflow(apps, switchCount, shortcutCount, spanMs) {
    const clusterKey = sortedAppsKey(apps);
    if (!clusterKey) return;
    if (!appWorkflows[clusterKey]) {
      appWorkflows[clusterKey] = { count: 0, switch_count: 0, shortcut_count: 0, total_span_ms: 0 };
    }
    appWorkflows[clusterKey].count++;
    appWorkflows[clusterKey].switch_count += Number(switchCount) || 0;
    appWorkflows[clusterKey].shortcut_count += Number(shortcutCount) || 0;
    appWorkflows[clusterKey].total_span_ms += Number(spanMs) || 0;
  }

  for (const line of lines) {
    let entry;
    try { entry = JSON.parse(line); } catch { continue; }

    const ts = entry.ts;
    const eventType = entry.type || "shortcut";
    const app = entry.app || "unknown";
    const layer = String(entry.layer || "0");

    if (!earliest || ts < earliest) earliest = ts;
    if (!latest || ts > latest) latest = ts;

    if (eventType === "mouse") {
      const btn = entry.button || "MB1";
      const cnt = entry.count || 1;
      if (!mouseClicks[btn]) mouseClicks[btn] = { count: 0, apps: {}, with_modifier: {} };
      mouseClicks[btn].count += cnt;
      mouseClicks[btn].apps[app] = (mouseClicks[btn].apps[app] || 0) + cnt;
      if (entry.modifier) {
        mouseClicks[btn].with_modifier[entry.modifier] = (mouseClicks[btn].with_modifier[entry.modifier] || 0) + cnt;
      }
      if (!mouseByLayer[btn]) mouseByLayer[btn] = {};
      mouseByLayer[btn][layer] = (mouseByLayer[btn][layer] || 0) + cnt;
      if (!byApp[app]) byApp[app] = { total: 0, shortcuts: {}, mouse_clicks: 0, scroll_total: 0 };
      byApp[app].mouse_clicks = (byApp[app].mouse_clicks || 0) + cnt;
      continue;
    }

    if (eventType === "scroll") {
      const dir = entry.direction || "down";
      const ticks = entry.ticks || 1;
      const exe = app.toLowerCase();
      if (!scrollEvents[exe]) scrollEvents[exe] = { up: 0, down: 0, left: 0, right: 0 };
      scrollEvents[exe][dir] = (scrollEvents[exe][dir] || 0) + ticks;
      scrollTotal += ticks;
      scrollByLayer[layer] = (scrollByLayer[layer] || 0) + ticks;
      if (!byApp[app]) byApp[app] = { total: 0, shortcuts: {}, mouse_clicks: 0, scroll_total: 0 };
      byApp[app].scroll_total = (byApp[app].scroll_total || 0) + ticks;
      continue;
    }

    if (eventType === "layer_session") {
      const sl = String(entry.layer || "0");
      if (!layerSessions[sl]) layerSessions[sl] = { count: 0, total_duration_ms: 0, total_keys: 0, key_freq: {} };
      layerSessions[sl].count++;
      layerSessions[sl].total_duration_ms += entry.duration_ms || 0;
      const kp = entry.keys_pressed || [];
      layerSessions[sl].total_keys += kp.length;
      for (const k of kp) {
        layerSessions[sl].key_freq[k] = (layerSessions[sl].key_freq[k] || 0) + 1;
      }
      continue;
    }

    if (eventType === "app_focus") {
      const prevApp = entry.prev_app || "";
      const dur = entry.prev_duration_ms || 0;
      if (prevApp) {
        appFocusTime[prevApp] = (appFocusTime[prevApp] || 0) + dur;
      }
      continue;
    }

    if (eventType === "shortcut_sequence") {
      addShortcutSequence(
        entry.from,
        entry.to,
        entry.gap_ms,
        entry.from_app || app,
        entry.to_app || app,
        entry.same_app,
      );
      continue;
    }

    if (eventType === "shortcut_workflow_window") {
      addShortcutWorkflow(entry.keys || [], entry.apps || [], entry.layers || [], entry.span_ms || 0);
      continue;
    }

    if (eventType === "app_transition") {
      addAppSequence(entry.from_app, entry.to_app, entry.prev_duration_ms || 0);
      continue;
    }

    if (eventType === "app_workflow_window") {
      addAppWorkflow(entry.apps || [], entry.switch_count || 0, entry.shortcut_count || 0, entry.span_ms || 0);
      continue;
    }

    if (eventType === "modifier_tap") {
      const key = entry.key || "unknown";
      modifierTaps[key] = (modifierTaps[key] || 0) + 1;
      continue;
    }

    if (eventType === "mouse_session") {
      mouseSessions.push({
        started_with: entry.started_with || "MB1",
        keyboard_shortcuts: entry.keyboard_shortcuts || [],
        duration_ms: entry.duration_ms || 0,
        app: app,
        layer: layer,
      });
      continue;
    }

    if (eventType === "correction") {
      corrections.push({
        attempted: entry.attempted || "",
        corrected_with: entry.corrected_with || "",
        gap_ms: entry.gap_ms || 0,
        app: app,
      });
      continue;
    }

    if (eventType === "modifier_error") {
      modifierErrors.push({
        modifier: entry.modifier || "",
        followed_by: entry.followed_by || "",
        duration_ms: entry.duration_ms || 0,
        app: app,
      });
      continue;
    }

    if (eventType === "layer_transition") {
      layerTransitions.push({
        from: entry.from || "0",
        to: entry.to || "0",
        method: entry.method || "unknown",
        duration_ms: entry.duration_ms || 0,
        keys_on_target: entry.keys_on_target || 0,
        app: app,
      });
      continue;
    }

    if (eventType === "layer_bounce") {
      layerBounces.push({
        layers: entry.layers || [],
        total_ms: entry.total_ms || 0,
        app: app,
      });
      continue;
    }

    if (eventType === "layer_sticky") {
      layerSticky.push({
        layer: entry.layer || "0",
        duration_ms: entry.duration_ms || 0,
        keys_pressed: entry.keys_pressed || 0,
        app: app,
      });
      continue;
    }

    if (eventType === "shortcut_retry") {
      const rkeys = entry.keys || "unknown";
      if (!shortcutRetries[rkeys]) shortcutRetries[rkeys] = { count: 0, gaps: [] };
      shortcutRetries[rkeys].count++;
      shortcutRetries[rkeys].gaps.push(entry.gap_ms || 0);
      continue;
    }

    if (eventType === "shortcut_noop_hint") {
      const akeys = entry.attempted || "unknown";
      if (!shortcutNoopHints[akeys]) shortcutNoopHints[akeys] = { count: 0, followed_by: {} };
      shortcutNoopHints[akeys].count++;
      const fb = entry.followed_by || "unknown";
      shortcutNoopHints[akeys].followed_by[fb] = (shortcutNoopHints[akeys].followed_by[fb] || 0) + 1;
      continue;
    }

    if (eventType === "app_context") {
      const ctx = entry.context || "general";
      if (!byContext[ctx]) byContext[ctx] = { count: 0, apps: {}, inferred_from: {} };
      byContext[ctx].count++;
      byContext[ctx].apps[app] = (byContext[ctx].apps[app] || 0) + 1;
      const inf = entry.inferred_from || "unknown";
      byContext[ctx].inferred_from[inf] = (byContext[ctx].inferred_from[inf] || 0) + 1;
      continue;
    }

    if (eventType === "typing_counter") {
      const tkeys = entry.keys || "unknown";
      const tcount = entry.count || 1;
      if (!shortcuts[tkeys]) shortcuts[tkeys] = { count: 0, apps: {}, by_layer: {} };
      shortcuts[tkeys].count += tcount;
      shortcuts[tkeys].apps[app] = (shortcuts[tkeys].apps[app] || 0) + tcount;
      shortcuts[tkeys].by_layer[layer] = (shortcuts[tkeys].by_layer[layer] || 0) + tcount;
      if (!byApp[app]) byApp[app] = { total: 0, shortcuts: {}, mouse_clicks: 0, scroll_total: 0 };
      byApp[app].total += tcount;
      byApp[app].shortcuts[tkeys] = (byApp[app].shortcuts[tkeys] || 0) + tcount;
      continue;
    }

    if (eventType === "raw_completion_key") {
      addRawCompletionUsage(entry.keys || "", entry.base_key || "", entry.count || 1, app, layer);
      continue;
    }

    // shortcut, functional, layer_key — all handled as key events
    const rawKeys = entry.keys;
    if (!rawKeys) continue;
    const keys = normalizeKeys(rawKeys);
    const repeatCount = entry.repeat_count || 1;

    if (!shortcuts[keys]) shortcuts[keys] = { count: 0, apps: {}, by_layer: {} };
    shortcuts[keys].count += repeatCount;
    shortcuts[keys].apps[app] = (shortcuts[keys].apps[app] || 0) + repeatCount;
    shortcuts[keys].by_layer[layer] = (shortcuts[keys].by_layer[layer] || 0) + repeatCount;

    if (!byApp[app]) byApp[app] = { total: 0, shortcuts: {}, mouse_clicks: 0, scroll_total: 0 };
    byApp[app].total += repeatCount;
    byApp[app].shortcuts[keys] = (byApp[app].shortcuts[keys] || 0) + repeatCount;

    byLayer[layer] = (byLayer[layer] || 0) + repeatCount;

    // Hand tracking
    if (entry.hand) {
      const h = entry.hand;
      if (!byHand[h]) byHand[h] = { count: 0, shortcuts: {} };
      byHand[h].count += repeatCount;
      byHand[h].shortcuts[keys] = (byHand[h].shortcuts[keys] || 0) + repeatCount;
    }

    // First-seen tracking (from logger sidecar)
    if (entry.first_seen) {
      if (!shortcutFirstSeen[keys] || entry.first_seen < shortcutFirstSeen[keys]) {
        shortcutFirstSeen[keys] = entry.first_seen;
      }
    }

    // Sequence confidence: collect raw gap_ms per sequence for CV computation
    if (entry.prev && entry.gap_ms) {
      addShortcutSequence(entry.prev, keys, entry.gap_ms, app, app, true);
    }

    // Sequence tracking (existing)
    // Handled by addShortcutSequence above.

    if (entry.held_ms && entry.held_ms >= 500) {
      if (!holdHeavy[keys]) holdHeavy[keys] = { count: 0, total_hold_ms: 0 };
      holdHeavy[keys].count++;
      holdHeavy[keys].total_hold_ms += entry.held_ms;
    }

    // Chain detection: variable-length sliding windows (2, 3, 4, 5 events)
    const evt = { keys, ts, app, layer };
    shortcutEventLog.push(evt);
    chainBuffer.push(evt);
    if (chainBuffer.length > 5) chainBuffer.shift();
    for (let winSize = 2; winSize <= chainBuffer.length; winSize++) {
      const win = chainBuffer.slice(chainBuffer.length - winSize);
      const spanMs = new Date(win[winSize - 1].ts) - new Date(win[0].ts);
      if (spanMs > 10000) continue;
      const chainKey = win.map(e => e.keys).join(" -> ");
      if (!chains[chainKey]) chains[chainKey] = { count: 0, total_ms: 0 };
      chains[chainKey].count++;
      chains[chainKey].total_ms += spanMs;
      if (winSize >= 3) {
        addShortcutWorkflow(
          win.map(e => e.keys),
          win.map(e => e.app),
          win.map(e => e.layer),
          spanMs,
        );
      }
    }
  }

  for (const [, seq] of Object.entries(sequences)) {
    seq.avg_gap_ms = Math.round(seq.total_gap_ms / seq.count);
    delete seq.total_gap_ms;
  }

  let periodDays = 0;
  if (earliest && latest) {
    periodDays = Math.max(1, Math.round((new Date(latest) - new Date(earliest)) / 86400000));
  }

  for (const [, s] of Object.entries(shortcuts)) {
    s.per_day = Math.round((s.count / Math.max(periodDays, 1)) * 10) / 10;
  }

  for (const [, a] of Object.entries(byApp)) {
    a.top = Object.entries(a.shortcuts).sort((x, y) => y[1] - x[1]).slice(0, 10).map(e => e[0]);
    delete a.shortcuts;
  }

  // Finalize layer sessions
  for (const [, ls] of Object.entries(layerSessions)) {
    ls.avg_duration_ms = Math.round(ls.total_duration_ms / Math.max(ls.count, 1));
    delete ls.total_duration_ms;
    ls.avg_keys_per_session = Math.round((ls.total_keys / Math.max(ls.count, 1)) * 10) / 10;
    ls.common_keys = Object.entries(ls.key_freq).sort((a, b) => b[1] - a[1]).slice(0, 10).map(e => e[0]);
    delete ls.key_freq;
  }

  // Layer switch activations: how many times each layer was activated
  const layerSwitchActivations = {};
  for (const [layerNum, ls] of Object.entries(layerSessions)) {
    layerSwitchActivations[layerNum] = ls.count;
  }

  // Finalize hold-heavy keys
  for (const [, h] of Object.entries(holdHeavy)) {
    h.avg_hold_ms = Math.round(h.total_hold_ms / Math.max(h.count, 1));
    delete h.total_hold_ms;
  }

  // ── App time tracking from charybdis_events.jsonl ──
  const appTime = {};  // exe -> seconds in foreground
  const appEvents = parseEvents();
  for (let i = 0; i < appEvents.length - 1; i++) {
    const cur = appEvents[i];
    const next = appEvents[i + 1];
    const exe = extractExe(cur.activeApp || "");
    if (!exe) continue;
    const dt = (new Date(next.updatedAt) - new Date(cur.updatedAt)) / 1000;
    if (dt > 0 && dt < 300) { // cap at 5 min per event gap (idle filter)
      appTime[exe] = (appTime[exe] || 0) + dt;
    }
  }

  // Merge app_focus time into appTime (more accurate when available)
  for (const [exe, ms] of Object.entries(appFocusTime)) {
    const key = exe.toLowerCase();
    const sec = Math.round(ms / 1000);
    if (sec > 0) appTime[key] = (appTime[key] || 0) + sec;
  }

  // ── Finalize chains ──
  for (const [, c] of Object.entries(chains)) {
    c.avg_total_ms = Math.round(c.total_ms / Math.max(c.count, 1));
    delete c.total_ms;
  }

  // ── Workflow detection: repeated subsequences of 3-5 shortcuts ──
  const workflows = {};
  for (let winSize = 3; winSize <= 5; winSize++) {
    for (let i = 0; i <= shortcutEventLog.length - winSize; i++) {
      const window = shortcutEventLog.slice(i, i + winSize);
      const spanMs = new Date(window[winSize - 1].ts) - new Date(window[0].ts);
      if (spanMs > 15000) continue;
      const apps = new Set(window.map(e => e.app));
      if (apps.size > 1) continue;
      const wfKey = window.map(e => e.keys).join(" -> ");
      if (!workflows[wfKey]) workflows[wfKey] = { count: 0, apps: {} };
      workflows[wfKey].count++;
      const app = window[0].app;
      workflows[wfKey].apps[app] = (workflows[wfKey].apps[app] || 0) + 1;
    }
  }
  // Filter: only keep workflows with count >= 3
  for (const key of Object.keys(workflows)) {
    if (workflows[key].count < 3) delete workflows[key];
  }

  // Merge legacy detected workflows into the new shortcut_workflows structure.
  for (const [wfKey, wfData] of Object.entries(workflows)) {
    if (!shortcutWorkflows[wfKey]) {
      shortcutWorkflows[wfKey] = {
        count: 0,
        total_span_ms: 0,
        apps: {},
        app_count: 0,
        layer_count: 0,
      };
    }
    shortcutWorkflows[wfKey].count += wfData.count || 0;
    for (const [appName, count] of Object.entries(wfData.apps || {})) {
      shortcutWorkflows[wfKey].apps[appName] = (shortcutWorkflows[wfKey].apps[appName] || 0) + count;
    }
    shortcutWorkflows[wfKey].app_count = Math.max(shortcutWorkflows[wfKey].app_count, Object.keys(wfData.apps || {}).length);
  }

  for (const [seqKey, seq] of Object.entries(shortcutSequences)) {
    const gaps = seq.gaps || [];
    const avg = gaps.length ? gaps.reduce((a, b) => a + b, 0) / gaps.length : 0;
    seq.avg_gap_ms = Math.round(avg);
    seq.p50_gap_ms = Math.round(percentile(gaps, 0.5));
    const appCoverage = Object.keys(seq.apps || {}).length;
    const consistency = gaps.length >= 2 && avg > 0
      ? Math.max(0.25, 1.0 - (Math.sqrt(gaps.reduce((sum, g) => sum + (g - avg) ** 2, 0) / gaps.length) / avg))
      : 0.5;
    const countConfidence = Math.min(1.0, seq.count / 5);
    const speedConfidence = avg > 0 ? Math.max(0.25, Math.min(1.0, 3000 / avg)) : 0.5;
    const crossAppBoost = seq.cross_app_count > 0 ? 1.1 : 1.0;
    seq.confidence = Math.round(Math.min(1.0, countConfidence * consistency * speedConfidence * crossAppBoost) * 100) / 100;
    seq.apps = Object.fromEntries(Object.entries(seq.apps || {}).sort((a, b) => b[1] - a[1]));
    delete seq.gaps;
    if (!sequences[seqKey]) sequences[seqKey] = { count: seq.count, avg_gap_ms: seq.avg_gap_ms };
  }

  for (const [, wf] of Object.entries(shortcutWorkflows)) {
    wf.avg_span_ms = Math.round((wf.total_span_ms || 0) / Math.max(wf.count || 0, 1));
    wf.apps = Object.fromEntries(Object.entries(wf.apps || {}).sort((a, b) => b[1] - a[1]));
    delete wf.total_span_ms;
  }

  for (const [, seq] of Object.entries(appSequences)) {
    seq.avg_prev_duration_ms = Math.round((seq.total_prev_duration_ms || 0) / Math.max(seq.count || 0, 1));
    delete seq.total_prev_duration_ms;
  }

  for (const [, wf] of Object.entries(appWorkflows)) {
    wf.avg_span_ms = Math.round((wf.total_span_ms || 0) / Math.max(wf.count || 0, 1));
    delete wf.total_span_ms;
  }

  // ── Blind spot analysis ──
  // Cross-reference: apps the user spends time in vs shortcuts they actually use
  const blindSpots = analyzeBlindSpots(shortcuts, appTime, periodDays);

  // Build by_layer_shortcut: {keys -> {layer -> count}} for layer-aware fitness
  const byLayerShortcut = {};
  for (const [keys, data] of Object.entries(shortcuts)) {
    if (data.by_layer && Object.keys(data.by_layer).length > 0) {
      byLayerShortcut[keys] = data.by_layer;
    }
  }
  const layerShortcuts = {};
  for (const [keys, byLayerCounts] of Object.entries(byLayerShortcut)) {
    for (const [layer, count] of Object.entries(byLayerCounts || {})) {
      const layerKey = String(layer);
      if (!layerShortcuts[layerKey]) {
        layerShortcuts[layerKey] = { total: 0, shortcuts: {} };
      }
      layerShortcuts[layerKey].total += count;
      layerShortcuts[layerKey].shortcuts[keys] = (
        layerShortcuts[layerKey].shortcuts[keys] || 0
      ) + count;
    }
  }
  for (const layerData of Object.values(layerShortcuts)) {
    const sorted = Object.entries(layerData.shortcuts).sort((a, b) => b[1] - a[1]);
    layerData.shortcuts = Object.fromEntries(sorted);
    layerData.top_shortcuts = sorted.slice(0, 20).map(([keys, count]) => ({ keys, count }));
  }

  // Finalize new optimizer v2 structures
  // Shortcut confidence: compute CV (coefficient of variation) for each sequence
  for (const [seqKey, gaps] of Object.entries(sequenceGaps)) {
    if (gaps.length < 3) continue;
    const avg = gaps.reduce((a, b) => a + b, 0) / gaps.length;
    const variance = gaps.reduce((sum, g) => sum + (g - avg) ** 2, 0) / gaps.length;
    const std = Math.sqrt(variance);
    const cv = avg > 0 ? std / avg : 0;
    if (!shortcutConfidence[seqKey]) shortcutConfidence[seqKey] = { avg_gap_ms: 0, cv: 0, sample_count: 0 };
    shortcutConfidence[seqKey].avg_gap_ms = Math.round(avg);
    shortcutConfidence[seqKey].cv = Math.round(cv * 100) / 100;
    shortcutConfidence[seqKey].sample_count = gaps.length;
  }

  // Aggregate mouse sessions: which keyboard shortcuts are commonly used with mouse
  const mouseSessionShortcuts = {};
  for (const ms of mouseSessions) {
    for (const sk of ms.keyboard_shortcuts) {
      if (!mouseSessionShortcuts[sk]) mouseSessionShortcuts[sk] = { count: 0, started_with: {} };
      mouseSessionShortcuts[sk].count++;
      const sb = ms.started_with || "MB1";
      mouseSessionShortcuts[sk].started_with[sb] = (mouseSessionShortcuts[sk].started_with[sb] || 0) + 1;
    }
  }

  const output = {
    timestamp: new Date().toISOString(),
    total_events: lines.length,
    period_days: periodDays,
    shortcuts,
    sequences,
    shortcut_sequences: shortcutSequences,
    by_app: byApp,
    by_layer: byLayer,
    by_layer_shortcut: byLayerShortcut,
    layer_shortcuts: layerShortcuts,
    raw_completion_keys: rawCompletionKeys,
    raw_completion_total: rawCompletionTotal,
    chains,
    workflows: shortcutWorkflows,
    shortcut_workflows: shortcutWorkflows,
    app_sequences: appSequences,
    app_workflows: appWorkflows,
    app_time_seconds: appTime,
    blind_spots: blindSpots,
    mouse_clicks: mouseClicks,
    mouse_by_layer: mouseByLayer,
    scroll_events: scrollEvents,
    scroll_total: scrollTotal,
    scroll_by_layer: scrollByLayer,
    layer_sessions: layerSessions,
    layer_switch_activations: layerSwitchActivations,
    hold_heavy: holdHeavy,
    modifier_taps: modifierTaps,
    // New optimizer v2 fields
    mouse_sessions: mouseSessions,
    mouse_session_shortcuts: mouseSessionShortcuts,
    corrections,
    modifier_errors: modifierErrors,
    layer_transitions: layerTransitions,
    layer_bounces: layerBounces,
    layer_sticky: layerSticky,
    shortcut_retries: shortcutRetries,
    shortcut_noop_hints: shortcutNoopHints,
    by_context: byContext,
    by_hand: byHand,
    shortcut_first_seen: shortcutFirstSeen,
    shortcut_confidence: shortcutConfidence,
  };

  writeBuild("usage_stats.json", output);

  const nBlind = blindSpots.length;
  return {
    success: true,
    output: { summary: `${lines.length} events, ${Object.keys(shortcuts).length} unique shortcuts, ${periodDays} days, ${nBlind} blind spots` },
    errors, warnings,
  };
}

function readFileBomSafe(filepath) {
  if (!fs.existsSync(filepath)) return null;
  let raw = fs.readFileSync(filepath, "utf-8");
  if (raw.charCodeAt(0) === 0xFEFF) raw = raw.slice(1);
  return raw;
}

function parseEvents() {
  const raw = readFileBomSafe(EVENTS_LOG);
  if (!raw) return [];
  return raw.split(/\r?\n/).filter(Boolean).map(line => {
    try { return JSON.parse(line); } catch { return null; }
  }).filter(Boolean);
}

function extractExe(activeAppStr) {
  // "msedge.exe - ZMK Studio..." => "msedge.exe"
  // "explorer.exe - " => "explorer.exe"
  const match = activeAppStr.match(/^(\S+\.exe)\b/i);
  if (match) return match[1].toLowerCase();
  // Non-exe entries like "Charybdis beacon listener" => skip
  return null;
}

// Map exe names to app names used in app_shortcut_scores.json
const EXE_TO_APP = {
  "msedge.exe": "Browser (Chrome/Edge)",
  "chrome.exe": "Browser (Chrome/Edge)",
  "code.exe": "Visual Studio Code",
  "explorer.exe": "File Explorer",
  "windowsterminal.exe": "Windows Terminal / PowerShell",
  "powershell.exe": "Windows Terminal / PowerShell",
  "pwsh.exe": "Windows Terminal / PowerShell",
  "excel.exe": "Microsoft Excel",
  "winword.exe": "Microsoft Word",
  "powerpnt.exe": "Microsoft PowerPoint",
  "outlook.exe": "Microsoft Outlook",
  "teams.exe": "Microsoft Teams",
  "ms-teams.exe": "Microsoft Teams",
  "discord.exe": "Discord",
  "taskmgr.exe": null, // no shortcut corpus
  "searchhost.exe": null,
};

function analyzeBlindSpots(usedShortcuts, appTime, periodDays) {
  // Load the shortcut corpus
  let scores;
  try { scores = JSON.parse(fs.readFileSync(SCORES_PATH, "utf-8")); } catch { return []; }

  // Shortcuts that are almost certainly used via direct keyboard (not AHK-routed)
  // and should not be flagged as blind spots just because they're not in the log
  // These shortcuts go directly from keyboard to OS/app — they never pass through
  // AHK's SendSafe, so absence from the log does NOT mean the user doesn't use them.
  // With the expanded logger, most shortcuts are now actually captured.
  // Only assume keys that Windows intercepts before AHK can see them.
  const ASSUMED_USED = new Set([
    "Ctrl+Alt+Delete",
  ]);
  // Single bare keys that aren't real "shortcuts" (Vimium-style, game keys)
  const BARE_KEY_RE = /^[a-zA-Z0-9]$/;

  // Build set of shortcuts the user has actually used (normalized keys)
  const usedKeys = new Set(Object.keys(usedShortcuts));

  // Build per-app used shortcuts (from the shortcut log app field)
  const usedByApp = {};
  for (const [keys, info] of Object.entries(usedShortcuts)) {
    for (const appExe of Object.keys(info.apps || {})) {
      const exe = appExe.toLowerCase();
      if (!usedByApp[exe]) usedByApp[exe] = new Set();
      usedByApp[exe].add(keys);
    }
  }

  const blindSpots = [];

  for (const app of scores.apps) {
    const appName = app.name;

    // Find which exe(s) map to this app
    const matchingExes = Object.entries(EXE_TO_APP)
      .filter(([, name]) => name === appName)
      .map(([exe]) => exe);

    // Calculate time spent in this app
    let timeInApp = 0;
    for (const exe of matchingExes) {
      timeInApp += appTime[exe] || 0;
    }

    // Collect shortcuts used in this specific app
    const appUsedKeys = new Set();
    for (const exe of matchingExes) {
      for (const k of (usedByApp[exe] || [])) appUsedKeys.add(k);
    }

    for (const shortcut of app.shortcuts) {
      const keys = shortcut.keys;
      const importance = shortcut.importance || 0;
      if (importance < 3.0) continue; // only flag important ones

      const isUsed = usedKeys.has(keys) || appUsedKeys.has(keys);
      if (isUsed) continue;

      // Skip shortcuts that are almost certainly used (direct keyboard, not AHK-routed)
      if (ASSUMED_USED.has(keys)) continue;
      // Skip bare single-character keys (Vimium-style, not real shortcuts)
      if (BARE_KEY_RE.test(keys)) continue;

      // Blind spot scoring:
      // - Higher importance = bigger blind spot
      // - More time in the app = more surprising the user doesn't use it
      // - Shortcuts with universal keys (Ctrl+C etc) may be used but not logged
      //   via AHK (only SendSafe-routed shortcuts are logged)
      const timeWeight = timeInApp > 0 ? Math.min(3.0, Math.log(1 + timeInApp / 60)) : 0.5;
      const score = importance * timeWeight;

      if (score < 2.0) continue; // filter noise

      blindSpots.push({
        app: appName,
        keys,
        action: shortcut.action || "",
        importance,
        time_in_app_min: Math.round(timeInApp / 60),
        blind_spot_score: Math.round(score * 10) / 10,
        reason: timeInApp > 300
          ? `User spent ${Math.round(timeInApp/60)}min in ${appName} but never used ${keys} (${shortcut.action})`
          : `High-importance shortcut not observed in usage data`,
      });
    }
  }

  // Sort by blind spot score descending
  blindSpots.sort((a, b) => b.blind_spot_score - a.blind_spot_score);
  return blindSpots;
}

module.exports = { run, normalizeKeys };

if (require.main === module) {
  const result = run({});
  console.log("Usage aggregation:", result.output.summary);
}
