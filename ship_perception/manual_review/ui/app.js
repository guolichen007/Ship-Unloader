"use strict";

const $ = id => document.getElementById(id);
const viewport = $("viewport"), canvas = $("map"), svg = $("overlay");
const token = location.hash.slice(1) || sessionStorage.getItem("mhr-token");
if (location.hash) {
  sessionStorage.setItem("mhr-token", token);
  history.replaceState(null, "", "/");
}
const state = { generation: 0, layerGeneration: 0, mode: "draw", zoom: 1,
  pan: [0, 0], rect: null, drag: null, candidate: null, annotation: crypto.randomUUID(),
  parent: 0, targets: {}, request: null, busy: false, enabled: false };
let m, image;
const ns = "http://www.w3.org/2000/svg";
const identity = ["session_id", "frame_instance_id", "view_id", "manifest_sha256",
  "wire_sha256", "semantic_sha256", "frame_id"];
const localIdentity = () => Object.fromEntries(identity.map(k => [k, m[k]]));
const status = text => { $("status").textContent = text; };
const changes = () => {
  if (state.busy && state.request && m && state.enabled) {
    api("/api/cancel", {...localIdentity(), request_id: state.request.request_id}).catch(() => {});
  }
  state.generation++;
  state.candidate = null;
  $("evidence").replaceChildren();
  buttons();
};
async function api(path, data) {
  const headers = { "X-MHR-Token": token || "" };
  const options = { headers };
  if (data !== undefined) {
    headers["Content-Type"] = "application/json";
    options.method = "POST";
    options.body = JSON.stringify(data);
  }
  const response = await fetch(path, options);
  if (!response.ok) {
    const error = await response.json();
    throw Error(error.error);
  }
  return response;
}
function camera() {
  const {width, height} = viewport.getBoundingClientRect();
  const scale = Math.min(width / m.nx, height / m.ny) * state.zoom;
  return {width, height, scale,
    x: (width - m.nx * scale) / 2 + state.pan[0],
    y: (height - m.ny * scale) / 2 + state.pan[1]};
}
const project = ([u, v], c) => [c.x + u * c.scale, c.y + v * c.scale];
const unproject = ([x, y], c) => [(x - c.x) / c.scale, (y - c.y) / c.scale];
const world = ([u, v]) => [m.x0 + u * m.s, m.y0 + (m.ny - v) * m.s];
const pixel = ([x, y]) => [(x - m.x0) / m.s, m.ny - (y - m.y0) / m.s];
const inMap = ([u, v]) => u >= 0 && v >= 0 && u <= m.nx && v <= m.ny;
function vertices() {
  if (!state.rect) return [];
  const {center: [u, v], w, h} = state.rect;
  const theta = Number($("angle").value) * Math.PI / 180;
  const a = Math.cos(theta), b = Math.sin(theta);
  return [[-w,-h],[w,-h],[w,h],[-w,h]].map(([x,y]) => [u+a*x-b*y, v+b*x+a*y]);
}
function drawPolygon(points, cameraState, color, dash, label) {
  const css = points.map(p => project(p, cameraState));
  const node = document.createElementNS(ns, "polygon");
  node.setAttribute("points", css.map(p => p.join(",")).join(" "));
  node.setAttribute("stroke", color);
  node.setAttribute("stroke-dasharray", dash);
  node.dataset.label = label;
  svg.append(node);
  const text = document.createElementNS(ns, "text");
  text.setAttribute("x", css[0][0] + 4);
  text.setAttribute("y", css[0][1] - 6);
  text.setAttribute("fill", color);
  text.textContent = label;
  svg.append(text);
}
function buttons() {
  const rough = m && state.rect && vertices().every(inMap) &&
    2 * Math.min(state.rect.w, state.rect.h) * m.s >= 1 &&
    2 * Math.max(state.rect.w, state.rect.h) * m.s <= 250;
  $("match").disabled = !rough || state.busy;
  $("manual").disabled = !state.enabled || !rough || !state.candidate || state.busy;
  $("accept").disabled = !state.enabled || state.candidate?.match_state !== "MATCHED_REVIEW" || state.busy;
  $("undo").disabled = !state.enabled || !state.parent || state.busy;
  $("export").disabled = !state.enabled || !state.parent || state.busy;
}
function draw() {
  if (!m) return;
  const c = camera(), dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(c.width * dpr));
  const height = Math.max(1, Math.round(c.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width; canvas.height = height;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(width / c.width, 0, 0, height / c.height, 0, 0);
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.imageSmoothingEnabled = false;
  if (image) ctx.drawImage(image, c.x, c.y, m.nx*c.scale, m.ny*c.scale);
  svg.replaceChildren();
  if ($("showAuto").checked) {
    Object.entries(state.targets).forEach(([id, h]) =>
      drawPolygon(h.xy.map(pixel), c, "#94bbff", "3 5", id));
  }
  const rough = vertices();
  if (rough.length && $("showRough").checked)
    drawPolygon(rough, c, rough.every(inMap) ? "#ffce73" : "#ff8480", "9 5", "人工粗框");
  const saved = state.saved?.[state.annotation];
  if (saved) drawPolygon(saved.xy.map(pixel), c, "#c5a2ff", "12 4 2 4", "已保存修订");
  if (state.candidate?.geometry && $("showCandidate").checked)
    drawPolygon(state.candidate.geometry.xy.map(pixel), c, "#73e9ac", "", "Raw 候选");
  $("geometry").textContent = rough.length ?
    rough.map((p,i) => "P"+(i+1)+" "+world(p).map(x=>x.toFixed(3)).join(", ")).join("\n")+
    "\n长 "+(2*Math.max(state.rect.w,state.rect.h)*m.s).toFixed(3)+" m · 宽 "+
    (2*Math.min(state.rect.w,state.rect.h)*m.s).toFixed(3)+" m" : "";
  buttons();
}
async function layer() {
  const generation = ++state.layerGeneration;
  try {
    const response = await api("/layer/" + $("layer").value + ".png");
    const bitmap = await createImageBitmap(await response.blob());
    if (generation !== state.layerGeneration) { bitmap.close(); return; }
    if (image) image.close();
    image = bitmap;
    draw();
  } catch (error) { status("图层加载失败 · " + error.message); }
}
function pointer(event) {
  const b = viewport.getBoundingClientRect();
  return [event.clientX - b.left, event.clientY - b.top];
}
viewport.addEventListener("pointerdown", event => {
  if (!m || state.drag || event.button !== 0) return;
  const p = pointer(event), at = unproject(p, camera());
  if (state.mode === "draw" && !inMap(at)) return;
  viewport.setPointerCapture(event.pointerId);
  state.drag = { id: event.pointerId, mode: state.mode, p, at, pan: [...state.pan] };
  if (state.mode === "draw") {
    changes();
    state.rect = {center: at, w: 0, h: 0};
    $("angle").value = 0;
    status("正在圈选");
    draw();
  }
});
viewport.addEventListener("pointermove", event => {
  const d = state.drag;
  if (!d || d.id !== event.pointerId) return;
  const p = pointer(event);
  if (d.mode === "pan") state.pan = d.pan.map((v, i) => v+p[i]-d.p[i]);
  else {
    const b = unproject(p, camera());
    state.rect = {center: b.map((v,i) => (v+d.at[i])/2),
      w: Math.abs(b[0]-d.at[0])/2, h: Math.abs(b[1]-d.at[1])/2};
  }
  draw();
});
viewport.addEventListener("lostpointercapture", () => { state.drag = null; });
viewport.addEventListener("wheel", event => {
  if (!m) return;
  event.preventDefault();
  const p = pointer(event), anchor = unproject(p, camera());
  state.zoom = Math.max(1, Math.min(60, state.zoom * Math.exp(-event.deltaY * .002)));
  const next = project(anchor, camera());
  state.pan = state.pan.map((v,i) => v+p[i]-next[i]);
  draw();
}, {passive: false});
function cancel() {
  changes(); state.rect = null; state.drag = null; draw(); status("已取消圈选");
}
for (const mode of ["draw","pan"]) $(mode).onclick = () => {
  state.mode = mode;
  for (const key of ["draw","pan"]) $(key).setAttribute("aria-pressed", String(key === mode));
};
$("reset").onclick = () => { state.zoom = 1; state.pan = [0,0]; draw(); };
$("angle").oninput = () => { changes(); draw(); };
$("layer").onchange = layer;
for (const key of ["showAuto","showRough","showCandidate"]) $(key).onchange = draw;
$("cancel").onclick = cancel;
window.addEventListener("keydown", event => { if (event.key === "Escape") cancel(); });
new ResizeObserver(draw).observe(viewport);
$("target").onchange = () => {
  cancel();
  state.annotation = $("target").value.replace(/^annotation:/, "") || crypto.randomUUID();
  state.parent = state.heads?.[state.annotation] || 0;
  draw();
};
$("match").onclick = async () => {
  const generation = state.generation, verts = vertices();
  const target = Object.hasOwn(state.targets, $("target").value) ? $("target").value : "";
  const request = { ...localIdentity(), schema: "manual_selection.1",
    request_id: crypto.randomUUID(), annotation_id: state.annotation,
    expected_parent_revision: state.parent, operation: target ? "REFINE" : "CREATE",
    target_opening_id: target || null, target_vessel_id: state.targets[target]?.vessel_id || null,
    vertices_uv: verts, client_xy: verts.map(world) };
  state.busy = true; state.request = request; buttons(); status("正在验证");
  try {
    const result = await (await api(state.enabled ? "/api/match" : "/api/selection", request)).json();
    if (generation !== state.generation || result.request_id !== request.request_id || result.view_id !== m.view_id) return;
    state.candidate = result;
    showEvidence(result);
    const states = {MATCHED_REVIEW:"已生成复核候选", PARTIAL:"证据不足", AMBIGUOUS:"边界有歧义",
      NO_MATCH:"未匹配", TIMEOUT:"计算超时", CANCELLED:"已取消", BUDGET_EXCEEDED:"超过计算预算"};
    status(result.state === "VALID" ? "坐标验证通过" :
      (states[result.match_state || result.execution_state] || "匹配异常") +
      (result.reason ? " · " + result.reason : ""));
  } catch (error) {
    if (generation === state.generation) status("操作失败 · " + error.message);
  } finally { state.busy = false; state.request = null; draw(); }
};
function showEvidence(result) {
  $("evidence").replaceChildren();
  for (const side of result.sides || []) {
    const panel = document.createElement("details"), label = document.createElement("summary");
    label.textContent = side.side_id + " · " + side.role + " · " + side.reason;
    panel.append(label);
    for (const mode of side.modes) {
      const details = document.createElement("p");
      details.textContent = "偏移 "+mode.offset_m.toFixed(3)+" m · 训练断面 "+
        mode.train.length+" · 留出 "+mode.heldout.length+" · "+mode.reason;
      panel.append(details);
    }
    $("evidence").append(panel);
  }
  if (result.geometry) {
    const z = document.createElement("p");
    z.textContent = result.geometry.z.map((v,i) => "P"+(i+1)+" Z "+
      (v === null ? "未观测" : v.toFixed(3)+" m")).join(" · ");
    $("evidence").append(z);
  }
}
async function revision(action, extra = {}) {
  const annotation = state.annotation;
  state.busy = true; buttons();
  try {
    const result = await (await api("/api/" + action, {...localIdentity(),
      mutation_id: crypto.randomUUID(), annotation_id: state.annotation,
      expected_parent_revision: state.parent, ...extra})).json();
    if (action === "export") {
      for (const [extension, type, content] of [["json","application/json",result.json],
        ["csv","text/csv;charset=utf-8",result.csv]]) {
        const link = document.createElement("a");
        link.href = URL.createObjectURL(new Blob([content], {type}));
        link.download = "人工船舱复核." + extension;
        link.click();
        setTimeout(() => URL.revokeObjectURL(link.href), 5000);
      }
      status("已导出复核结果");
    } else {
      state.heads[annotation] = result.revision;
      state.saved[annotation] = result.geometry;
      if (!state.targets[annotation]) {
        const value = "annotation:" + annotation;
        if (![...$("target").options].some(opt => opt.value === value)) {
          const opt = document.createElement("option");
          opt.value = value; opt.textContent = "人工修订 " + annotation.slice(0,8);
          $("target").append(opt);
        }
        if (annotation === state.annotation) $("target").value = value;
      }
      if (annotation === state.annotation) {
        state.parent = result.revision;
        changes(); status("已保存修订 " + result.revision);
      }
    }
  } catch (error) { status("修订失败 · " + error.message); }
  finally { state.busy = false; draw(); }
}
function confirmResult(action) {
  const r = state.candidate;
  if (!r || (action === "manual" && !window.confirm("仅保存人工粗框，所有 Z 标为未观测。是否继续？"))) return;
  revision(action, {request_id: r.request_id, suggestion_id: r.suggestion_id,
    candidate_id: r.candidate_id, geometry_digest: action === "manual" ? r.rough_digest : r.geometry_digest});
}
$("accept").onclick = () => confirmResult("accept");
$("manual").onclick = () => confirmResult("manual");
$("undo").onclick = () => revision("undo");
$("export").onclick = () => revision("export");
(async () => {
  try {
    const view = await (await api("/api/view")).json();
    m = view.manifest; state.targets = view.targets; state.enabled = view.review_enabled || false;
    state.heads = view.heads || {}; state.saved = view.saved_geometry || {};
    for (const [id, target] of Object.entries(state.targets)) {
      const opt = document.createElement("option");
      opt.value = id; opt.textContent = target.vessel_id + " / " + id;
      $("target").append(opt);
    }
    for (const id of Object.keys(state.heads)) {
      if (state.targets[id]) continue;
      const opt = document.createElement("option");
      opt.value = "annotation:" + id; opt.textContent = "人工修订 " + id.slice(0,8);
      $("target").append(opt);
    }
    await layer();
    status(m.overlay_status === "INVALID_OR_WRONG_FRAME" ?
      "自动叠加身份无效，已禁用。可新建船舱。" : "拖动绘制粗框，可调整角度。");
  } catch (error) { status("无法打开视图 · " + error.message); }
})();
