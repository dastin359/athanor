"""Launch multiple ARC solver web instances in parallel with a dashboard.

Usage:
    python -m athanor.web_demo.batch_launcher \
        --tasks e87109e9 271d71e2 28a6681f \
        --auto-start

    Opens dashboard at http://127.0.0.1:7860
    Each puzzle gets its own full web UI on ports 7870, 7871, ...
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
import urllib.request
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any


# ── Shared mutable state for dynamic puzzle management ──────────────────
class _BatchState:
    def __init__(self):
        self.lock = threading.Lock()
        self.instances: list[dict[str, Any]] = []
        self.procs: list[subprocess.Popen] = []
        self.auto_start: bool = False
        self.base_port: int = 7870

_state = _BatchState()


def _is_port_free(port: int) -> bool:
    """Check if a TCP port is available."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _find_free_port(start: int, count: int) -> list[int]:
    """Find `count` consecutive-ish free ports starting from `start`."""
    ports = []
    candidate = start
    while len(ports) < count and candidate < start + count * 10:
        if _is_port_free(candidate):
            ports.append(candidate)
        candidate += 1
    return ports


def _launch_instance(task_id: str, port: int) -> subprocess.Popen:
    """Launch a single web app instance as a subprocess."""
    env = os.environ.copy()
    env["PHOENIX_PROJECT_NAME"] = f"ARC_batch_{task_id}"
    cmd = [
        sys.executable, "-m", "athanor.web_demo",
        "--port", str(port),
        "--host", "127.0.0.1",
    ]
    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.DEVNULL,
        stderr=None,
    )
    return proc


def _wait_for_ready(port: int, timeout: float = 30.0) -> bool:
    """Poll until the instance responds on /api/status."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=2)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def _set_puzzle_via_ws(port: int, task_id: str):
    """Connect to an instance's WebSocket and set puzzle_path config (without starting)."""
    try:
        import websockets.sync.client
    except ImportError:
        return False
    try:
        with websockets.sync.client.connect(
            f"ws://127.0.0.1:{port}/ws", open_timeout=5, close_timeout=2,
        ) as ws:
            ws.recv(timeout=5)  # initial snapshot
            ws.send(json.dumps({"type": "config_update", "config": {"puzzle_path": task_id}}))
            time.sleep(0.3)  # let the server process it
        return True
    except Exception:
        return False


def _start_run_via_ws(port: int, task_id: str, config: dict[str, Any] | None = None):
    """Connect to an instance's WebSocket and send a start command."""
    try:
        import websockets
        import websockets.sync.client
    except ImportError:
        print(f"    ⚠ websockets package not installed — skipping auto-start for {task_id}")
        print(f"      Install with: pip install websockets")
        print(f"      Or start manually at http://127.0.0.1:{port}")
        return False

    start_config = {**(config or {}), "puzzle_path": task_id}
    try:
        with websockets.sync.client.connect(
            f"ws://127.0.0.1:{port}/ws",
            open_timeout=5,
            close_timeout=2,
        ) as ws:
            # Wait for initial snapshot
            ws.recv(timeout=5)
            # Send start
            ws.send(json.dumps({"type": "start", "config": start_config}))
            # Wait for run_state confirmation
            deadline = time.time() + 5
            while time.time() < deadline:
                raw = ws.recv(timeout=3)
                msg = json.loads(raw)
                if msg.get("type") == "run_state" and msg.get("running"):
                    return True
        return True
    except Exception as e:
        print(f"    ⚠ Failed to auto-start {task_id}: {e}")
        return False


def _add_puzzle_background(task_id: str, auto_start: bool):
    """Background thread: launch a new instance, wait, configure, optionally start."""
    label = task_id or "new"
    # Find a free port above all currently allocated ones
    with _state.lock:
        start = max((inst["port"] for inst in _state.instances), default=_state.base_port) + 1
    candidate = start
    port = None
    while candidate < start + 100:
        if _is_port_free(candidate):
            port = candidate
            break
        candidate += 1
    if port is None:
        print(f"  [add] ERROR: no free port for {label}")
        return

    # Use a unique key: task_id if given, otherwise "new_<port>"
    inst_id = task_id if task_id else f"new_{port}"
    inst = {"task_id": inst_id, "port": port, "launching": True}
    with _state.lock:
        _state.instances.append(inst)

    proc = _launch_instance(inst_id, port)
    with _state.lock:
        _state.procs.append(proc)
        inst["pid"] = proc.pid
    print(f"  [add] {label} → :{port} (pid {proc.pid})")

    if _wait_for_ready(port):
        if task_id:
            _set_puzzle_via_ws(port, task_id)
            if auto_start:
                _start_run_via_ws(port, task_id, config=getattr(_state, 'run_config', None))
        with _state.lock:
            inst["launching"] = False
        print(f"  [add] ✓ {label} ready on :{port}")
    else:
        with _state.lock:
            inst["launching"] = False
        print(f"  [add] ✗ {label} FAILED to start on :{port}")


def _build_dashboard_html(instances: list[dict[str, Any]], dashboard_port: int) -> str:
    """Generate dashboard HTML with a floating run-selector island over the iframe's header."""
    rows_json = json.dumps(instances)
    return textwrap.dedent(f"""\
    <!DOCTYPE html>
    <html><head>
    <title>ARC Batch Dashboard</title>
    <meta charset="utf-8">
    <style>
      * {{ margin: 0; padding: 0; box-sizing: border-box; }}
      html, body {{ width: 100%; height: 100%; overflow: hidden;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Inter, sans-serif;
        color: #fff;
        background: linear-gradient(180deg, #0a4a43, #063a35);
      }}

      /* ── Floating run-selector island (overlays iframe's green topbar) ── */
      .run-island {{
        position: fixed; z-index: 9999;
        top: 13px; left: 24px;
        max-width: max(280px, calc(100vw - 920px));
        display: flex; align-items: center; gap: 0;
        padding: 9px 0 9px 10px;
        border-radius: 30px;
        background: rgba(255, 255, 255, 0.09);
        border: 1px solid rgba(160, 241, 210, 0.18);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.08), 0 8px 24px rgba(0,0,0,0.28);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
      }}
      .run-scroll {{
        display: flex; align-items: center; gap: 8px;
        overflow-x: auto; overflow-y: hidden;
        padding: 0 14px 0 8px;
        flex: 1; min-width: 0;
      }}
      .run-scroll::-webkit-scrollbar {{ height: 4px; }}
      .run-scroll::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.2); border-radius: 2px; }}
      .run-btn {{
        border: 1px solid rgba(138, 227, 190, 0.10); background: rgba(255, 255, 255, 0.04);
        color: rgba(255, 255, 255, 0.75); font-size: 15px; font-weight: 700;
        border-radius: 20px; padding: 11px 18px;
        cursor: pointer; transition: all 0.22s ease; white-space: nowrap;
        flex-shrink: 0;
        display: flex; align-items: center; gap: 6px;
      }}
      .run-btn:hover {{
        background: rgba(255, 255, 255, 0.10);
        border-color: rgba(126, 233, 188, 0.35);
        color: #fff;
        box-shadow: 0 0 14px rgba(16, 185, 129, 0.18);
      }}
      .run-btn.active {{
        background: rgba(18, 174, 131, 0.32);
        border-color: rgba(120, 238, 192, 0.40);
        color: #fff;
        box-shadow: inset 0 0 0 1px rgba(226, 255, 244, 0.12), 0 0 16px rgba(16, 185, 129, 0.22);
      }}
      .close-x {{
        margin-left: 2px; opacity: 0; transition: opacity 0.15s;
        font-size: 14px; line-height: 1; padding: 0 2px; color: rgba(255,255,255,0.5);
      }}
      .close-x:hover {{ color: #f87171; opacity: 1 !important; }}
      .run-btn:hover .close-x {{ opacity: 0.7; }}
      .dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }}
      .dot.running {{ background: #fbbf24; animation: pulse 1.5s infinite; }}
      .dot.solved {{ background: #34d399; }}
      .dot.failed {{ background: #f87171; }}
      .dot.idle {{ background: #555; }}
      .dot.offline {{ background: #333; }}
      @keyframes pulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.4; }} }}

      .run-summary {{ display: none; }}

      /* ── Content area ── */
      .main {{ position: absolute; inset: 0; }}
      .main iframe {{ width: 100%; height: 100%; border: none; position: absolute; inset: 0; }}

      .overview {{
        padding: 110px 30px 30px;
        height: 100%; overflow: auto;
      }}
      table {{
        border-collapse: collapse; width: 100%; max-width: 900px;
        margin: 0 auto;
      }}
      th, td {{
        padding: 10px 16px;
        border: 1px solid rgba(138, 227, 190, 0.15);
        text-align: left; font-size: 14px;
      }}
      th {{
        background: rgba(12, 88, 77, 0.58);
        color: rgba(199, 243, 223, 0.8); font-weight: 600;
      }}
      tr:hover {{ background: rgba(12, 88, 77, 0.35); }}
      .status-running {{ color: #fbbf24; }}
      .status-solved {{ color: #34d399; font-weight: bold; }}
      .status-failed {{ color: #f87171; }}
      .status-idle {{ color: rgba(199, 243, 223, 0.4); }}
      .status-offline {{ color: rgba(199, 243, 223, 0.2); }}
      a {{ color: rgba(126, 233, 188, 0.9); text-decoration: none; }}
      a:hover {{ text-decoration: underline; }}

      .add-wrap {{ position: relative; display: flex; align-items: center; gap: 6px; }}
      .ac-list {{
        position: absolute; top: 100%; left: 0; margin-top: 4px;
        min-width: 180px; max-height: 240px; overflow-y: auto;
        background: rgba(12, 50, 42, 0.96); border: 1.5px solid rgba(132, 231, 196, 0.50);
        border-radius: 10px; box-shadow: 0 12px 32px rgba(0,0,0,0.5);
        backdrop-filter: blur(20px); z-index: 10000; padding: 4px 0;
      }}
      .ac-item {{
        padding: 7px 14px; font-size: 14px; color: #c8e6dc; cursor: pointer;
        transition: background 0.12s;
      }}
      .ac-item:hover, .ac-item.active {{ background: rgba(16, 185, 129, 0.20); }}
    </style>
    </head><body>

    <!-- Floating island — no header bar, just the pill -->
    <div class="run-island">
      <button class="run-btn" id="status-btn" onclick="switchTab('overview')">Status</button>
      <div class="run-scroll" id="tabs"></div>
      <button class="run-btn" id="add-btn" onclick="toggleAddForm()" title="Add puzzle" style="font-size:18px;padding:8px 14px;">+</button>
      <div class="add-wrap" id="add-form" style="display:none;"></div>
    </div>
    <div class="run-summary" id="summary"></div>
    <div class="main" id="main"></div>

    <script>
    const DASHBOARD_PORT = {dashboard_port};
    const instances = {rows_json};
    let activeTab = "overview";
    let statuses = {{}};
    let allTaskIds = [];
    let acIndex = -1;

    // Fetch task IDs from dashboard server for autocomplete
    (async function loadTaskIds() {{
      try {{
        const r = await fetch(`http://127.0.0.1:${{DASHBOARD_PORT}}/api/task-ids`);
        const ids = await r.json();
        if (Array.isArray(ids) && ids.length) allTaskIds = ids;
      }} catch(e) {{}}
    }})();

    function statusClass(s) {{
      if (!s) return "offline";
      if (s.running && !s.paused) return "running";
      if (s.status === "idle" && s.usage && s.usage.total_requests > 0) {{
        if (s.test_accuracy != null && s.test_accuracy < 1.0) return "failed";
        return "solved";
      }}
      if (s.status === "idle") return "idle";
      return "idle";
    }}

    let addFormVisible = false;
    let addInputValue = "";
    function renderTabs() {{
      const tabs = document.getElementById("tabs");
      // Preserve input value before rebuilding
      const existingInput = document.getElementById("add-input");
      if (existingInput) addInputValue = existingInput.value;

      // Update pinned Status button
      const statusBtn = document.getElementById("status-btn");
      if (statusBtn) statusBtn.className = `run-btn ${{activeTab === 'overview' ? 'active' : ''}}`;
      // Build scrollable puzzle tabs
      let html = '';
      for (const inst of instances) {{
        const s = statuses[inst.task_id];
        const cls = inst.launching ? "idle" : statusClass(s);
        const active = activeTab === inst.task_id ? "active" : "";
        const suffix = inst.launching ? " …" : (s && s.iteration ? ` i${{s.iteration}}` : "");
        const puzzleId = s && s.puzzle_id ? s.puzzle_id : "";
        const label = puzzleId ? puzzleId.substring(0,8) : (inst.task_id.startsWith("new_") ? "new" : inst.task_id.substring(0,8));
        html += `<button class="run-btn ${{active}}" onclick="switchTab('${{inst.task_id}}')">
          <span class="dot ${{cls}}"></span>${{label}}${{suffix}}<span class="close-x" onclick="event.stopPropagation(); removePuzzle('${{inst.task_id}}')" title="Close">&times;</span></button>`;
      }}
      tabs.innerHTML = html;
      // Update pinned add-form
      const addForm = document.getElementById("add-form");
      if (addForm) {{
        if (addFormVisible) {{
          addForm.style.display = "";
          addForm.innerHTML = `
            <input id="add-input" type="text" placeholder="task_id" autocomplete="off" value="${{addInputValue.replace(/"/g, '&quot;')}}"
              style="background:rgba(255,255,255,0.12);border:1px solid rgba(160,241,210,0.25);border-radius:14px;padding:8px 12px;color:#fff;font-size:14px;width:160px;outline:none;"
              oninput="onAddInput()" onkeydown="onAddKeydown(event)">
            <button class="run-btn" onclick="addPuzzle()" style="padding:8px 14px;font-size:14px;">Go</button>
            <button class="run-btn" onclick="addEmpty()" style="padding:8px 14px;font-size:14px;opacity:0.6;">Empty</button>
            <div id="ac-dropdown" class="ac-list" style="display:none;"></div>`;
          const inp = document.getElementById("add-input");
          if (inp) {{
            inp.focus();
            inp.setSelectionRange(inp.value.length, inp.value.length);
            if (inp.value) onAddInput();
          }}
        }} else {{
          addForm.style.display = "none";
          addForm.innerHTML = "";
        }}
      }}
    }}
    function toggleAddForm() {{
      addFormVisible = !addFormVisible;
      if (!addFormVisible) addInputValue = "";
      renderTabs();
    }}
    async function addPuzzle(taskIdOverride) {{
      const inp = document.getElementById("add-input");
      const taskId = taskIdOverride != null ? taskIdOverride : (inp && inp.value || "").trim();
      if (taskIdOverride == null && !taskId) return; // Go button requires input
      try {{
        const r = await fetch(`http://127.0.0.1:${{DASHBOARD_PORT}}/api/add-puzzle`, {{
          method: "POST",
          headers: {{"Content-Type": "application/json"}},
          body: JSON.stringify({{task_id: taskId, auto_start: !!taskId}}),
        }});
        const data = await r.json();
        if (r.ok) {{
          addFormVisible = false;
          addInputValue = "";
          await pollAll();
        }} else {{
          alert(data.error || "Failed to add puzzle");
        }}
      }} catch(e) {{
        alert("Error: " + e.message);
      }}
    }}
    async function addEmpty() {{
      await addPuzzle("");
    }}
    async function removePuzzle(taskId) {{
      const s = statuses[taskId];
      const isRunning = s && s.running && !s.paused;
      if (isRunning) {{
        if (!confirm(`${{taskId}} is currently running. Stop and remove it?`)) return;
      }}
      try {{
        const r = await fetch(`http://127.0.0.1:${{DASHBOARD_PORT}}/api/remove-puzzle`, {{
          method: "POST",
          headers: {{"Content-Type": "application/json"}},
          body: JSON.stringify({{task_id: taskId}}),
        }});
        if (r.ok) {{
          const idx = instances.findIndex(i => i.task_id === taskId);
          if (idx >= 0) instances.splice(idx, 1);
          delete statuses[taskId];
          // Remove iframe if exists
          const iframe = document.getElementById("iframe-" + taskId);
          if (iframe) iframe.remove();
          if (activeTab === taskId) switchTab("overview");
          else {{ renderTabs(); renderMain(); }}
        }} else {{
          const data = await r.json();
          alert(data.error || "Failed to remove puzzle");
        }}
      }} catch(e) {{
        alert("Error: " + e.message);
      }}
    }}
    function onAddInput() {{
      const inp = document.getElementById("add-input");
      const dd = document.getElementById("ac-dropdown");
      if (!inp || !dd) return;
      const q = inp.value.trim().toLowerCase();
      acIndex = -1;
      if (!q || !allTaskIds.length) {{ dd.style.display = "none"; return; }}
      const existing = new Set(instances.map(i => i.task_id));
      const matches = allTaskIds.filter(t => t.toLowerCase().startsWith(q) && !existing.has(t)).slice(0, 12);
      if (!matches.length) {{ dd.style.display = "none"; return; }}
      dd.innerHTML = matches.map((t, i) =>
        `<div class="ac-item" onmousedown="selectAc('${{t}}')">${{t}}</div>`
      ).join("");
      dd.style.display = "block";
    }}
    function onAddKeydown(e) {{
      const dd = document.getElementById("ac-dropdown");
      if (!dd || dd.style.display === "none") {{
        if (e.key === "Enter") addPuzzle();
        return;
      }}
      const items = dd.querySelectorAll(".ac-item");
      if (e.key === "ArrowDown") {{
        e.preventDefault();
        acIndex = Math.min(acIndex + 1, items.length - 1);
        items.forEach((el, i) => el.classList.toggle("active", i === acIndex));
      }} else if (e.key === "ArrowUp") {{
        e.preventDefault();
        acIndex = Math.max(acIndex - 1, 0);
        items.forEach((el, i) => el.classList.toggle("active", i === acIndex));
      }} else if (e.key === "Enter") {{
        e.preventDefault();
        if (acIndex >= 0 && acIndex < items.length) {{
          selectAc(items[acIndex].textContent);
        }} else {{
          addPuzzle();
        }}
      }} else if (e.key === "Escape") {{
        dd.style.display = "none";
        acIndex = -1;
      }}
    }}
    function selectAc(taskId) {{
      const inp = document.getElementById("add-input");
      if (inp) inp.value = taskId;
      const dd = document.getElementById("ac-dropdown");
      if (dd) dd.style.display = "none";
      acIndex = -1;
      addPuzzle();
    }}

    function renderMain() {{
      const main = document.getElementById("main");
      if (activeTab === "overview") {{
        main.querySelectorAll("iframe").forEach(f => f.style.display = "none");
        let overview = document.getElementById("overview-panel");
        if (!overview) {{
          overview = document.createElement("div");
          overview.id = "overview-panel";
          overview.className = "overview";
          main.appendChild(overview);
        }}
        overview.style.display = "block";
        let rows = "";
        let totInput = 0, totOutput = 0, totCost = 0, totInCost = 0, totOutCost = 0;
        for (const inst of instances) {{
          const s = statuses[inst.task_id];
          const cls = "status-" + statusClass(s);
          const statusLabel = s ? (s.running ? (s.paused ? "PAUSED" : "RUNNING") : s.status.toUpperCase()) : "OFFLINE";
          const iter = s ? (s.iteration || 0) : "—";
          const turn = s ? (s.turn || 0) : "—";
          const inp = s && s.usage ? s.usage.input_tokens : 0;
          const out = s && s.usage ? s.usage.output_tokens : 0;
          const cst = s && s.usage ? s.usage.total_cost : 0;
          const inCst = s && s.usage ? (s.usage.input_cost || 0) : 0;
          const outCst = s && s.usage ? (s.usage.output_cost || 0) : 0;
          totInput += inp; totOutput += out; totCost += cst; totInCost += inCst; totOutCost += outCst;
          // Accuracy info (available when idle/done)
          let accCell = "—";
          if (s && s.test_accuracy != null) {{
            const pct = (s.test_accuracy * 100).toFixed(0) + "%";
            const detail = s.test_correct_count != null && s.test_total != null
              ? ` (${{s.test_correct_count}}/${{s.test_total}})`
              : "";
            if (s.test_accuracy >= 1.0) {{
              accCell = `<span style="color:#34d399;font-weight:700;">${{pct}}${{detail}}</span>`;
            }} else {{
              const failed = s.test_solved_indices != null && s.test_total != null
                ? Array.from({{length: s.test_total}}, (_, i) => i).filter(i => !s.test_solved_indices.includes(i))
                : [];
              const failNote = failed.length ? ` <span style="color:#f87171;font-size:12px;">fail: ${{failed.join(",")}}</span>` : "";
              accCell = `<span style="color:#fbbf24;">${{pct}}${{detail}}</span>${{failNote}}`;
            }}
          }}
          rows += `<tr>
            <td><a href="#" onclick="switchTab('${{inst.task_id}}'); return false;">${{s && s.puzzle_id ? s.puzzle_id : inst.task_id}}</a></td>
            <td class="${{cls}}">${{statusLabel}}</td>
            <td>${{iter}}</td><td>${{turn}}</td>
            <td>${{accCell}}</td>
            <td>${{inp.toLocaleString()}}</td><td>${{out.toLocaleString()}}</td>
            <td>${{(inp+out).toLocaleString()}}</td>
            <td>$${{inCst.toFixed(2)}}</td><td>$${{outCst.toFixed(2)}}</td>
            <td>$${{cst.toFixed(2)}}</td>
            <td><a href="http://127.0.0.1:${{inst.port}}" target="_blank">Open</a></td>
          </tr>`;
        }}
        rows += `<tr style="border-top:2px solid rgba(138,227,190,0.40);font-weight:700;">
          <td colspan="5" style="text-align:right;">Total</td>
          <td>${{totInput.toLocaleString()}}</td><td>${{totOutput.toLocaleString()}}</td>
          <td>${{(totInput+totOutput).toLocaleString()}}</td>
          <td>$${{totInCost.toFixed(2)}}</td><td>$${{totOutCost.toFixed(2)}}</td>
          <td>$${{totCost.toFixed(2)}}</td><td></td>
        </tr>`;
        overview.innerHTML = `<table>
          <thead><tr><th>Puzzle</th><th>Status</th><th>Iter</th><th>Turn</th><th>Accuracy</th><th>Input Tok</th><th>Output Tok</th><th>Total Tok</th><th>Input $</th><th>Output $</th><th>Total $</th><th>Direct</th></tr></thead>
          <tbody>${{rows}}</tbody></table>`;
      }} else {{
        const inst = instances.find(i => i.task_id === activeTab);
        if (inst) {{
          const overviewEl = document.getElementById("overview-panel");
          if (overviewEl) overviewEl.style.display = "none";
          main.querySelectorAll("iframe").forEach(f => f.style.display = "none");
          const iframeId = "iframe-" + inst.task_id;
          let iframe = document.getElementById(iframeId);
          if (inst.launching) {{
            // Instance still booting — show placeholder instead of broken iframe
            if (iframe) iframe.style.display = "none";
            let ph = document.getElementById("ph-" + inst.task_id);
            if (!ph) {{
              ph = document.createElement("div");
              ph.id = "ph-" + inst.task_id;
              ph.style.cssText = "display:flex;align-items:center;justify-content:center;width:100%;height:100%;position:absolute;inset:0;color:#8fbab5;font-size:1.1rem;";
              ph.textContent = "Starting instance on port " + inst.port + " …";
              main.appendChild(ph);
            }}
            ph.style.display = "flex";
          }} else {{
            const ph = document.getElementById("ph-" + inst.task_id);
            if (ph) ph.style.display = "none";
            if (!iframe) {{
              iframe = document.createElement("iframe");
              iframe.id = iframeId;
              iframe.src = `http://127.0.0.1:${{inst.port}}`;
              main.appendChild(iframe);
            }}
            iframe.style.display = "block";
          }}
        }}
      }}
    }}

    function switchTab(id) {{
      activeTab = id;
      renderTabs();
      renderMain();
    }}

    function updateSummary() {{
      const total = instances.length;
      const running = instances.filter(i => statusClass(statuses[i.task_id]) === "running").length;
      const solved = instances.filter(i => statusClass(statuses[i.task_id]) === "solved").length;
      document.getElementById("summary").textContent =
        `${{running}} running · ${{solved}} done · ${{total}} total`;
    }}

    async function pollAll() {{
      // Merge dynamically added instances from server
      try {{
        const r = await fetch(`http://127.0.0.1:${{DASHBOARD_PORT}}/api/instances`);
        const serverInst = await r.json();
        for (const si of serverInst) {{
          if (!instances.find(i => i.task_id === si.task_id)) {{
            instances.push(si);
          }} else {{
            // Update launching status
            const local = instances.find(i => i.task_id === si.task_id);
            if (local) local.launching = si.launching;
          }}
        }}
      }} catch(e) {{}}
      // Poll status for each instance
      for (const inst of instances) {{
        try {{
          const r = await fetch(`http://127.0.0.1:${{inst.port}}/api/status`);
          statuses[inst.task_id] = await r.json();
        }} catch(e) {{
          statuses[inst.task_id] = null;
        }}
      }}
      renderTabs();
      updateSummary();
      renderMain();
    }}

    renderTabs();
    renderMain();
    pollAll();
    setInterval(pollAll, 3000);
    </script>
    </body></html>
    """)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the batch-launcher CLI arguments on `parser`.

    Exposed so the top-level `athanor` CLI can reuse this
    argument schema for its default `web` subcommand.
    """
    parser.add_argument("--tasks", nargs="*", default=[], help="Task IDs to solve (optional — launch empty dashboard if omitted; add more interactively via the + button)")
    parser.add_argument("--base-port", type=int, default=7870, help="First instance port (default: 7870)")
    parser.add_argument("--dashboard-port", type=int, default=7860, help="Dashboard port (default: 7860)")
    parser.add_argument("--auto-start", action="store_true", help="Automatically start solves on launch")
    parser.add_argument("--no-code-first-turn", action="store_true", default=False, help="Disable run_code tools on first turn (semi-CoT mode)")
    parser.add_argument("--thinking-effort", type=str, default="medium", choices=["low", "medium", "high", "max"], help="Main thinking effort (default: medium)")
    parser.add_argument("--reflection-thinking-effort", type=str, default="max", choices=["low", "medium", "high", "max"], help="Self-reflection thinking effort (default: max)")
    parser.add_argument("--compression-thinking-effort", type=str, default="max", choices=["low", "medium", "high", "max"], help="Context compression thinking effort (default: max)")
    parser.add_argument("--first-turn-thinking-effort", type=str, default="high", choices=["low", "medium", "high", "max"], help="First turn thinking effort when --no-code-first-turn (default: high)")
    parser.add_argument("--reflector-model", type=str, default=None, help="Reflector model (default: server default)")


def run(args: argparse.Namespace) -> int:
    """Run the batch launcher with parsed CLI args. Returns an exit code."""
    _state.auto_start = args.auto_start
    _state.base_port = args.base_port
    # Build run config from CLI flags
    _run_config: dict[str, Any] = {
        "semi_cot_first_turn": args.no_code_first_turn,
        "semi_cot_thinking_effort": args.first_turn_thinking_effort,
        "thinking_effort": args.thinking_effort,
        "reflection_thinking_effort": args.reflection_thinking_effort,
        "compression_thinking_effort": args.compression_thinking_effort,
    }
    if args.reflector_model:
        _run_config["reflector_model"] = args.reflector_model
    _state.run_config = _run_config
    instances = _state.instances  # alias
    procs = _state.procs

    # Find free ports for dashboard + all instances
    needed = len(args.tasks) + 1  # +1 for dashboard
    free_ports = _find_free_port(args.dashboard_port, needed)
    if len(free_ports) < needed:
        print(f"ERROR: Could only find {len(free_ports)} free ports (need {needed})")
        sys.exit(1)

    dashboard_port = free_ports[0]
    instance_ports = free_ports[1:]

    for i, task_id in enumerate(args.tasks):
        instances.append({"task_id": task_id, "port": instance_ports[i]})

    # Launch all instances
    print(f"Launching {len(instances)} instances...")
    for inst in instances:
        proc = _launch_instance(inst["task_id"], inst["port"])
        procs.append(proc)
        inst["pid"] = proc.pid
        print(f"  {inst['task_id']} → :{inst['port']} (pid {proc.pid})")

    # Wait for all to be ready
    print("Waiting for instances to start...")
    for inst in instances:
        if _wait_for_ready(inst["port"]):
            print(f"  ✓ {inst['task_id']} ready on :{inst['port']}")
        else:
            print(f"  ✗ {inst['task_id']} FAILED to start on :{inst['port']}")

    # Set puzzle_path on each instance
    for inst in instances:
        _set_puzzle_via_ws(inst["port"], inst["task_id"])

    # Auto-start if requested
    if args.auto_start:
        print("Auto-starting solves...")
        for inst in instances:
            ok = _start_run_via_ws(inst["port"], inst["task_id"], config=_run_config)
            if ok:
                print(f"  ✓ {inst['task_id']} started")

    # Build and serve dashboard
    dashboard_html = _build_dashboard_html(instances, dashboard_port).encode("utf-8")

    # Pre-load task IDs for autocomplete (same logic as app.py)
    try:
        from ..data import list_tasks
    except ImportError:
        try:
            from athanor.data import list_tasks
        except ImportError:
            list_tasks = None

    _task_ids_cache: list[str] = []
    if list_tasks is not None:
        try:
            _task_ids_cache = list_tasks(split="public_eval")
        except Exception:
            pass

    class DashboardHandler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/api/task-ids":
                data = json.dumps(_task_ids_cache).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
            elif self.path == "/api/instances":
                with _state.lock:
                    data = json.dumps(_state.instances).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(dashboard_html)

        def do_POST(self):
            if self.path == "/api/add-puzzle":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length)) if length else {}
                task_id = str(body.get("task_id", "")).strip()
                auto_start = bool(body.get("auto_start", _state.auto_start))
                # task_id can be empty — launches an empty instance for later use
                with _state.lock:
                    if task_id and any(i["task_id"] == task_id for i in _state.instances):
                        self._json(409, {"error": f"{task_id} already exists"})
                        return
                threading.Thread(
                    target=_add_puzzle_background,
                    args=(task_id, auto_start),
                    daemon=True,
                ).start()
                self._json(202, {"status": "launching", "task_id": task_id})
            elif self.path == "/api/remove-puzzle":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length)) if length else {}
                task_id = str(body.get("task_id", "")).strip()
                if not task_id:
                    self._json(400, {"error": "task_id required"})
                    return
                with _state.lock:
                    inst = next((i for i in _state.instances if i["task_id"] == task_id), None)
                    if not inst:
                        self._json(404, {"error": f"{task_id} not found"})
                        return
                    port = inst["port"]
                    pid = inst.get("pid")
                    _state.instances.remove(inst)
                    # Kill the matching subprocess by pid
                    if pid:
                        for proc in _state.procs:
                            try:
                                if proc.pid == pid:
                                    proc.kill()
                                    break
                            except Exception:
                                pass
                    # Remove procs that are no longer alive
                    _state.procs = [p for p in _state.procs if p.poll() is None]
                print(f"  [remove] {task_id} on :{port} killed")
                self._json(200, {"status": "removed", "task_id": task_id})
            else:
                self._json(404, {"error": "not found"})

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def _json(self, code, data):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        def log_message(self, *_args):
            pass  # Suppress request logs

    # Use threaded server so POST doesn't block GET
    from http.server import ThreadingHTTPServer
    server = ThreadingHTTPServer(("127.0.0.1", dashboard_port), DashboardHandler)

    print(f"\n{'='*50}")
    print(f"  Dashboard: http://127.0.0.1:{dashboard_port}")
    print(f"{'='*50}")
    for inst in instances:
        print(f"  {inst['task_id']}: http://127.0.0.1:{inst['port']}")
    print(f"\n  Ctrl+C to stop all instances")

    def _cleanup(*_args):
        print("\nShutting down...")
        with _state.lock:
            for proc in _state.procs:
                try:
                    proc.kill()
                except Exception:
                    pass
        os._exit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    server.serve_forever()
    return 0
