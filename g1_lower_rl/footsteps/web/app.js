"use strict";

// 按固定元素标识获取页面控件
const byId = (name) => document.getElementById(name);
const palette = { left: "#007f78", right: "#d76550", ink: "#25302e", muted: "#717976", grid: "#dfe5dc", amber: "#aa730a" };
const modes = { walking: "行走", stopping: "收步中", standing: "站立", starting: "起步中" };
let current = null;
let session = null;
let samples = [];
let running = false;
let chain = Promise.resolve();
let advancePending = false;
let lastTick = performance.now();
let accumulator = 0;
let chartMode = "frequency";
let coordinateFrame = "local";
let frequencyMode = "random";
let activeConfig = {};
let selectedId = null;
const camera = { x: 0.3, y: 0, scale: 300, zoom: 300, follow: true };
let hitTargets = [];

// 更新按钮的 Unicode 符号，并同步悬停说明和无障碍名称
function icon(button, symbol, label) {
  button.querySelector(".symbol").textContent = symbol;
  button.title = label;
  button.setAttribute("aria-label", label);
}

// 切换预览播放状态并清除待推进时间，不改变管理器的行走或站立模式
function setRunning(value) {
  running = value;
  accumulator = 0;
  lastTick = performance.now();
  icon(byId("play"), value ? "Ⅱ" : "▶", value ? "暂停播放" : "播放");
  byId("step").disabled = value;
}

// 串行发送会话请求，避免控制指令和推进请求互相越过
function enqueue(path, data = {}, onSuccess = accept) {
  const task = chain.then(async () => {
    const response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session, ...data }),
      signal: AbortSignal.timeout(10000),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
    byId("error").hidden = true;
    byId("connection").textContent = "本地已连接";
    onSuccess(result);
    return result;
  });
  chain = task.catch((error) => {
    setRunning(false);
    byId("error").textContent = `预览暂停：${error.message}`;
    byId("error").hidden = false;
    byId("connection").textContent = "请求失败";
  });
  return chain;
}

// 接收新命令并保留容量内的曲线采样，再刷新界面
function accept(data) {
  current = data;
  session = data.session;
  samples.push(...data.samples);
  if (samples.length > 9000) samples.splice(0, samples.length - 9000);
  render();
}

// 使用已应用或新提交的配置重置会话，并恢复默认显示状态
function reset(config = activeConfig) {
  setRunning(false);
  return enqueue("/api/reset", { config }, (data) => {
    activeConfig = config;
    samples = [];
    selectedId = null;
    camera.follow = true;
    camera.zoom = Math.min(340, byId("map").clientWidth * .8);
    frequencyMode = "random";
    byId("direction").value = 0;
    byId("direction-slider").value = 0;
    byId("heading").value = 0;
    byId("heading-slider").value = 0;
    for (const name of ["frequency-target", "frequency-slider"]) {
      byId(name).min = data.frequency_range[0];
      byId(name).max = data.frequency_range[1];
      byId(name).value = data.frequency.toFixed(2);
    }
    accept(data);
    updateFrequencyMode();
  });
}

// 将当前后端快照同步到指标、控制状态、四步表和三块画布
function render() {
  if (!current) return;
  const seconds = current.time;
  byId("clock").textContent = `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${(seconds % 60).toFixed(2).padStart(5, "0")}`;
  byId("frequency").textContent = current.frequency.toFixed(3);
  byId("cadence").textContent = (current.frequency * 2).toFixed(3);
  byId("phase").textContent = (current.phase * 180 / Math.PI).toFixed(1);
  byId("mode").textContent = modes[current.mode] || current.mode;
  byId("mode").dataset.mode = current.mode;
  byId("contact-left").classList.toggle("on", current.contacts[0]);
  byId("contact-right").classList.toggle("on", current.contacts[1]);
  byId("stop").disabled = ["standing", "stopping"].includes(current.mode);
  byId("start").disabled = current.mode !== "standing";
  byId("pending").textContent = current.mode === "stopping" ? "等待收步 / 双支撑" : "";
  byId("automation").checked = current.automatic;
  byId("seed-label").textContent = `SEED ${current.seed}`;
  byId("distance").textContent = `${current.landings.length} 次计划落地`;
  byId("frequency-min-label").textContent = `${current.frequency_range[0].toFixed(2)} Hz`;
  byId("frequency-max-label").textContent = `${current.frequency_range[1].toFixed(2)} Hz`;
  byId("random-rate").textContent = current.random_rate === null ? "--" : `${current.random_rate.toFixed(3)} Hz/s`;
  byId("distance-cap").textContent = `${current.effective_distance_max.toFixed(3)} m`;
  byId("width-cap").textContent = `${current.effective_width_range.map(value => value.toFixed(3)).join(" - ")} m`;
  byId("follow").classList.toggle("active", camera.follow);
  byId("follow").setAttribute("aria-pressed", String(camera.follow));
  if (current.automatic) {
    for (const field of ["direction", "heading"]) {
      if (document.activeElement !== byId(field) && document.activeElement !== byId(`${field}-slider`)) {
        byId(field).value = Math.round(current[field]);
        byId(`${field}-slider`).value = Math.round(current[field]);
      }
    }
  }
  document.querySelectorAll("[data-direction]").forEach((button) => button.classList.toggle("selected", Number(button.dataset.direction) === Math.round(current.direction)));
  const poses = coordinateFrame === "local" ? current.footsteps : current.footsteps_w;
  byId("queue").replaceChildren(...poses.map((pose, index) => {
    const row = document.createElement("tr");
    row.dataset.side = index < 2 ? "0" : "1";
    row.classList.toggle("highlight", selectedId === current.future_ids[index]);
    for (const value of [["L1", "L2", "R1", "R2"][index], `#${current.future_ids[index]}`, pose[0].toFixed(3), pose[1].toFixed(3), (pose[2] * 180 / Math.PI).toFixed(1)]) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    row.onclick = () => selectFoot(current.future_ids[index], pose, index < 2 ? 0 : 1, true);
    return row;
  }));
  byId("events").replaceChildren(...current.events.toReversed().slice(0, 12).map((event) => {
    const item = document.createElement("li");
    const time = document.createElement("time");
    time.textContent = `${event.time.toFixed(2)}s`;
    const text = document.createElement("span");
    const labels = { reset: "重置预览", stop: "请求站立", start: "请求起步", automation: event.enabled ? "启用自动指令" : "关闭自动指令", ...modes };
    text.textContent = event.kind === "direction" ? `方向 ${event.direction.toFixed(0)}° / 朝向 ${event.heading.toFixed(0)}°` : event.kind === "frequency" ? (event.target === null ? "随机慢变频率" : `目标 f = ${event.target.toFixed(2)} Hz`) : labels[event.kind] || event.kind;
    item.append(time, text);
    return item;
  }));
  drawMap();
  drawChart();
  drawCompass();
}

// 按设备像素比调整画布并清空，返回使用 CSS 像素坐标的绘图上下文
function surface(id) {
  const canvas = byId(id);
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(devicePixelRatio || 1, 2);
  if (canvas.width !== Math.round(rect.width * ratio) || canvas.height !== Math.round(rect.height * ratio)) {
    canvas.width = Math.round(rect.width * ratio);
    canvas.height = Math.round(rect.height * ratio);
  }
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, rect.width, rect.height);
  return { context, width: rect.width, height: rect.height };
}

// 在同一世界比例尺下绘制历史脚印、当前支撑、四步预览和共同参考系
function drawMap() {
  const { context, width, height } = surface("map");
  context.fillStyle = "#f1f4ef";
  context.fillRect(0, 0, width, height);
  if (!current) return;
  camera.scale = camera.zoom;
  if (camera.follow) {
    const poses = [...current.supports, ...current.footsteps_w];
    camera.x = poses.reduce((sum, pose) => sum + pose[0], 0) / poses.length;
    camera.y = poses.reduce((sum, pose) => sum + pose[1], 0) / poses.length;
    const spanX = Math.max(...poses.map((pose) => pose[0])) - Math.min(...poses.map((pose) => pose[0]));
    const spanY = Math.max(...poses.map((pose) => pose[1])) - Math.min(...poses.map((pose) => pose[1]));
    camera.scale = Math.min(camera.zoom, Math.max(25, (width - 75) / (spanX + .4)), Math.max(25, (height - 180) / (spanY + .4)));
  }
  const centerY = height / 2 + 28;
  // 将世界米坐标投影为画布像素，世界 Y 轴与屏幕纵轴方向相反
  const project = (pose) => [width / 2 + (pose[0] - camera.x) * camera.scale, centerY - (pose[1] - camera.y) * camera.scale];
  const spacing = camera.scale > 90 ? 0.25 : camera.scale > 40 ? 0.5 : 1;
  context.lineWidth = 1;
  context.strokeStyle = palette.grid;
  context.beginPath();
  const lowX = camera.x - width / 2 / camera.scale;
  const lowY = camera.y - height / camera.scale;
  for (let worldX = Math.floor(lowX / spacing) * spacing; worldX < lowX + width / camera.scale + spacing; worldX += spacing) {
    const screenX = project([worldX, 0])[0];
    context.moveTo(screenX, 0); context.lineTo(screenX, height);
  }
  for (let worldY = Math.floor(lowY / spacing) * spacing; worldY < lowY + 2 * height / camera.scale + spacing; worldY += spacing) {
    const screenY = project([0, worldY])[1];
    context.moveTo(0, screenY); context.lineTo(width, screenY);
  }
  context.stroke();
  const origin = project([0, 0]);
  context.strokeStyle = "#b5c0b6";
  context.beginPath();context.moveTo(origin[0], 0);context.lineTo(origin[0], height);context.moveTo(0, origin[1]);context.lineTo(width, origin[1]);context.stroke();
  context.fillStyle = palette.muted;context.font = "10px monospace";
  context.fillText("+X", width - 36, height / 2 + 24);context.fillText("+Y", width / 2 + 7, 18);
  const trail = current.landings;
  context.strokeStyle = "#a7b7a7";context.lineWidth = 1.2;context.beginPath();
  trail.forEach((landing, index) => { const point = project(landing.pose); index ? context.lineTo(...point) : context.moveTo(...point); });context.stroke();
  const future = current.footsteps_w.map((pose, index) => ({ pose, id: current.future_ids[index], side: index < 2 ? 0 : 1, slot: ["L1", "L2", "R1", "R2"][index] })).sort((first, second) => first.id - second.id);
  context.setLineDash([4, 5]);context.strokeStyle = "#7f8e82";context.beginPath();
  if (trail.length) context.moveTo(...project(trail.at(-1).pose));
  future.forEach((foot, index) => { const point = project(foot.pose); index || trail.length ? context.lineTo(...point) : context.moveTo(...point); });
  context.stroke();context.setLineDash([]);
  hitTargets = [];
  const labels = [];
  // 按模型足底几何绘制单个脚印，并收集标签和点选候选
  function foot(pose, side, kind, label, id) {
    const [screenX, screenY] = project(pose);
    const color = side ? palette.right : palette.left;
    context.save();context.translate(screenX, screenY);context.rotate(-pose[2]);
    const geometry = current.foot_geometry.feet[side];
    const length = geometry.length * camera.scale;
    const halfWidth = geometry.width * camera.scale / 2;
    const outline = new Path2D();
    for (const capsule of geometry.capsules) {
      const startX = capsule.start[0] * camera.scale, startY = -capsule.start[1] * camera.scale;
      const endX = capsule.end[0] * camera.scale, endY = -capsule.end[1] * camera.scale;
      const angle = Math.atan2(endY - startY, endX - startX);
      const radius = capsule.radius * camera.scale;
      outline.moveTo(startX + radius * Math.cos(angle - Math.PI / 2), startY + radius * Math.sin(angle - Math.PI / 2));
      outline.lineTo(endX + radius * Math.cos(angle - Math.PI / 2), endY + radius * Math.sin(angle - Math.PI / 2));
      outline.arc(endX, endY, radius, angle - Math.PI / 2, angle + Math.PI / 2);
      outline.lineTo(startX + radius * Math.cos(angle + Math.PI / 2), startY + radius * Math.sin(angle + Math.PI / 2));
      outline.arc(startX, startY, radius, angle + Math.PI / 2, angle + 3 * Math.PI / 2);
      outline.closePath();
    }
    context.strokeStyle = color;
    context.fillStyle = kind === "future" ? "#fafcf8" : color;
    context.lineWidth = id !== null && id === selectedId ? 3 : kind === "support" ? 1.5 : 1.2;
    context.setLineDash(kind === "future" ? [4, 3] : []);
    if (kind === "history") context.globalAlpha = 0.24;
    else context.stroke(outline);
    context.fill(outline);
    context.setLineDash([]);context.strokeStyle = kind === "support" ? "#ffffff" : color;context.beginPath();context.moveTo(-length * .12, 0);context.lineTo(length * .36, 0);context.lineTo(length * .23, -halfWidth * .36);context.moveTo(length * .36, 0);context.lineTo(length * .23, halfWidth * .36);context.stroke();context.restore();
    if (label) labels.push({ text: label, x: screenX, y: screenY, side, color });
    if (id !== null) hitTargets.push({ x: screenX, y: screenY, pose, side, id });
  }
  trail.forEach((landing) => foot(landing.pose, landing.side, "history", "", landing.id));
  current.supports.forEach((pose, side) => foot(pose, side, "support", "", null));
  future.forEach((item) => foot(item.pose, item.side, "future", `${item.slot} #${item.id}`, item.id));
  const placed = [];
  for (const label of labels) {
    context.font = "bold 10px monospace";
    const labelWidth = context.measureText(label.text).width + 10;
    const labelX = Math.max(labelWidth / 2 + 6, Math.min(width - labelWidth / 2 - 6, label.x));
    const offset = label.side ? 1 : -1;
    let labelY = label.y + offset * 29;
    while (placed.some((other) => Math.abs(other.x - labelX) < (other.width + labelWidth) / 2 && Math.abs(other.y - labelY) < 18)) labelY += offset * 19;
    placed.push({ x: labelX, y: labelY, width: labelWidth });
    context.strokeStyle = label.color;context.lineWidth = .6;context.beginPath();context.moveTo(label.x, label.y + offset * 12);context.lineTo(labelX, labelY - offset * 7);context.stroke();
    context.fillStyle = "#f1f4ef";context.fillRect(labelX - labelWidth / 2, labelY - 8, labelWidth, 15);
    context.fillStyle = label.color;context.textAlign = "center";context.fillText(label.text, labelX, labelY + 3);context.textAlign = "left";
  }
  const anchor = project(current.anchor_w);
  context.save();context.translate(...anchor);context.rotate(-current.anchor_w[2]);context.strokeStyle = palette.ink;context.lineWidth = 1.5;context.beginPath();context.moveTo(0, -22);context.lineTo(0, 0);context.lineTo(25, 0);context.stroke();context.restore();
  context.fillStyle = palette.ink;context.font = "10px monospace";context.fillText("A", anchor[0] - 12, anchor[1] + 18);
  const scaleLength = camera.scale > 100 ? .5 : 1;
  byId("scale").style.width = `${scaleLength * camera.scale}px`;
  byId("scale").textContent = `${scaleLength} m`;
}

// 按模拟时间绘制频率或相位曲线，以及左右脚计划接触条带
function drawChart() {
  const { context, width, height } = surface("chart");
  if (!current) return;
  const left = 39, right = width - 12, top = 13, bottom = height - 49;
  const duration = Number(byId("time-window").value);
  const end = Math.max(duration, current.time);
  const start = end - duration;
  const visible = samples.filter((sample) => sample.time >= start);
  const limit = chartMode === "frequency" ? Math.max(2, current.frequency_range[1] + .2) : 1.2;
  const lower = chartMode === "frequency" ? 0 : -1.2;
  // 将模拟时间映射到当前可见时间窗口的横坐标
  const screenX = (time) => left + (time - start) / duration * (right - left);
  // 将频率或相位值映射到对应纵轴范围
  const screenY = (value) => bottom - (value - lower) / (limit - lower) * (bottom - top);
  context.font = "9px monospace";context.fillStyle = palette.muted;context.strokeStyle = "#dce1dc";context.lineWidth = 1;
  const ticks = chartMode === "frequency" ? [0, .5, 1, 1.5, 2] : [-1, 0, 1];
  ticks.forEach((value) => {const position = screenY(value);context.fillText(value.toFixed(1), 8, position + 3);context.beginPath();context.moveTo(left, position);context.lineTo(right, position);context.stroke();});
  for (let index = 0; index <= 5; index++) {const time = start + index * duration / 5;const position = screenX(time);context.fillText(`${time.toFixed(0)}s`, Math.min(position, right - 18), height - 3);}
  // 绘制一个数据序列，空目标或不连续值处断开连线
  function line(getValue, color, dashed = false) {
    context.strokeStyle = color;context.lineWidth = 1.6;context.setLineDash(dashed ? [4, 4] : []);context.beginPath();let previous = null;
    for (const sample of visible) {
      const value = getValue(sample);
      if (value === null) {previous = null;continue;}
      const point = [screenX(sample.time), screenY(value)];
      if (!previous || Math.abs(value - previous.value) > (chartMode === "phase" ? 1.5 : 3)) context.moveTo(...point); else context.lineTo(...point);
      previous = { value };
    }
    context.stroke();context.setLineDash([]);
  }
  if (chartMode === "frequency") {line((sample) => sample.target, palette.amber, true);line((sample) => sample.frequency, palette.left);} else {line((sample) => Math.sin(sample.phase), palette.left);line((sample) => Math.cos(sample.phase), palette.right);}
  for (let side = 0; side < 2; side++) {
    const position = height - 34 + side * 10;
    context.fillStyle = palette.muted;context.fillText(side ? "R" : "L", 22, position + 5);
    context.fillStyle = "#e5eae4";context.fillRect(left, position, right - left, 6);
    context.fillStyle = side ? palette.right : palette.left;
    visible.forEach((sample, index) => {if (sample.contacts[side]) {const next = visible[index + 1]?.time ?? sample.time + .02;context.fillRect(screenX(sample.time), position, Math.max(1, screenX(Math.min(end, next)) - screenX(sample.time)), 6);}});
  }
  const cursor = screenX(current.time);context.strokeStyle = "#7d8980";context.lineWidth = 1;context.beginPath();context.moveTo(cursor, top);context.lineTo(cursor, height - 15);context.stroke();
}

// 用两个独立箭头展示世界系移动方向与脚掌朝向
function drawCompass() {
  const { context, width, height } = surface("compass");
  if (!current) return;
  const centerX = width / 2, centerY = height / 2, radius = Math.min(width, height) * .35;
  context.strokeStyle = "#d5ded4";context.lineWidth = 1;context.beginPath();context.arc(centerX, centerY, radius, 0, Math.PI * 2);context.moveTo(centerX - radius - 5, centerY);context.lineTo(centerX + radius + 5, centerY);context.moveTo(centerX, centerY - radius - 5);context.lineTo(centerX, centerY + radius + 5);context.stroke();
  // 从罗盘中心绘制给定角度、颜色和长度的方向箭头
  function arrow(degrees, color, length) {context.save();context.translate(centerX, centerY);context.rotate(-degrees * Math.PI / 180);context.strokeStyle = color;context.lineWidth = 2;context.beginPath();context.moveTo(0, 0);context.lineTo(length, 0);context.lineTo(length - 7, -4);context.moveTo(length, 0);context.lineTo(length - 7, 4);context.stroke();context.restore();}
  arrow(current.direction, palette.left, radius);arrow(current.heading, palette.right, radius * .7);
  context.fillStyle = palette.muted;context.font = "9px monospace";context.fillText("X", width - 7, centerY + 3);context.fillText("Y", centerX - 3, 8);
}

// 选中目标并显示其编号和坐标，local 表示数据来自当前表格坐标系
function selectFoot(id, pose, side, local = false) {
  selectedId = id;
  byId("selection").hidden = false;
  byId("selection").textContent = `${side ? "R" : "L"} #${id} · ${local ? (coordinateFrame === "local" ? "A" : "W") : "W"} (${pose[0].toFixed(3)}, ${pose[1].toFixed(3)}) m · ${(pose[2] * 180 / Math.PI).toFixed(1)}°`;
  render();
}

// 校验方向和朝向输入后发送指令，保留后端已经承诺的四步
function directionControl() {
  const direction = Number(byId("direction").value), heading = Number(byId("heading").value);
  if (!byId("direction").reportValidity() || !byId("heading").reportValidity()) return;
  enqueue("/api/control", { action: "direction", direction, heading });
}

// 同步随机或手动频率分段按钮与手动输入框的可用状态
function updateFrequencyMode() {
  document.querySelectorAll("[data-frequency-mode]").forEach((button) => button.classList.toggle("selected", button.dataset.frequencyMode === frequencyMode));
  byId("frequency-target").disabled = frequencyMode !== "manual";
  byId("frequency-slider").disabled = frequencyMode !== "manual";
}

// 播放与单步只推进模拟时间，停走按钮另行改变管理器状态
byId("play").onclick = () => {if (current) setRunning(!running);};
byId("step").onclick = () => enqueue("/api/advance", { frames: 1 });
byId("reset").onclick = () => reset();
byId("stop").onclick = () => enqueue("/api/control", { action: "stop" });
byId("start").onclick = () => enqueue("/api/control", { action: "start" });
byId("automation").onchange = (event) => enqueue("/api/control", { action: "automation", enabled: event.target.checked });
for (const name of ["direction", "heading"]) {
  byId(`${name}-slider`).oninput = (event) => {byId(name).value = event.target.value;};
  byId(`${name}-slider`).onchange = directionControl;
  byId(name).onchange = () => {byId(`${name}-slider`).value = byId(name).value;directionControl();};
}
document.querySelectorAll("[data-direction]").forEach((button) => button.onclick = () => {byId("direction").value = button.dataset.direction;byId("direction-slider").value = button.dataset.direction;directionControl();});
document.querySelectorAll("[data-frequency-mode]").forEach((button) => button.onclick = () => {
  frequencyMode = button.dataset.frequencyMode;
  updateFrequencyMode();
  enqueue("/api/control", { action: "frequency", value: frequencyMode === "random" ? null : Number(byId("frequency-target").value) });
});
byId("frequency-slider").oninput = (event) => {byId("frequency-target").value = event.target.value;};
byId("frequency-slider").onchange = () => enqueue("/api/control", { action: "frequency", value: Number(byId("frequency-target").value) });
byId("frequency-target").onchange = () => {if (byId("frequency-target").reportValidity()) {byId("frequency-slider").value = byId("frequency-target").value;enqueue("/api/control", { action: "frequency", value: Number(byId("frequency-target").value) });}};
// 参数作为完整配置提交，应用时重置会话而非中途改写已发布脚印
byId("settings-form").onsubmit = (event) => {event.preventDefault();reset(Object.fromEntries([...new FormData(event.target)].map(([key, value]) => [key, Number(value)])));};
// 下载后端保留范围内的 JSON 记录，并释放临时对象地址
byId("export").onclick = () => enqueue("/api/export", {}, (data) => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
  const link = document.createElement("a");link.href = url;link.download = `footsteps-seed${data.seed}-${data.state.time.toFixed(2)}s.json`;link.click();setTimeout(() => URL.revokeObjectURL(url), 1000);
});
document.querySelectorAll("[data-chart]").forEach((button) => button.onclick = () => {chartMode = button.dataset.chart;document.querySelectorAll("[data-chart]").forEach((other) => other.classList.toggle("selected", other === button));byId("chart-legend").textContent = chartMode === "frequency" ? "实线 f · 虚线目标" : "青色 sin · 珊瑚色 cos";drawChart();});
document.querySelectorAll("[data-frame]").forEach((button) => button.onclick = () => {coordinateFrame = button.dataset.frame;document.querySelectorAll("[data-frame]").forEach((other) => other.classList.toggle("selected", other === button));render();});
byId("time-window").onchange = drawChart;
// 调整用户期望缩放，实际显示比例仍受视口适配限制
function zoom(factor) {camera.zoom = Math.max(25, Math.min(650, camera.scale * factor));drawMap();}
byId("zoom-in").onclick = () => zoom(1.25);
byId("zoom-out").onclick = () => zoom(.8);
byId("follow").onclick = () => {camera.follow = !camera.follow;render();};
// 退出跟随并缩放到当前保留的完整轨迹范围
byId("fit").onclick = () => {
  if (!current) return;
  const poses = [...current.landings.map((item) => item.pose), ...current.footsteps_w, ...current.supports];
  const minX = Math.min(...poses.map((pose) => pose[0])), maxX = Math.max(...poses.map((pose) => pose[0]));
  const minY = Math.min(...poses.map((pose) => pose[1])), maxY = Math.max(...poses.map((pose) => pose[1]));
  camera.x = (minX + maxX) / 2;camera.y = (minY + maxY) / 2;camera.follow = false;
  const rect = byId("map").getBoundingClientRect();camera.zoom = Math.max(10, Math.min(250, (rect.width - 90) / (maxX - minX + .5), (rect.height - 160) / (maxY - minY + .5)));render();
};
// 记录拖动起点以平移视口，短点击则执行脚印命中检测
let drag = null;
byId("map").onpointerdown = (event) => {drag = { x: event.clientX, y: event.clientY, cameraX: camera.x, cameraY: camera.y };byId("map").setPointerCapture(event.pointerId);};
byId("map").onpointermove = (event) => {if (!drag) return;const deltaX = event.clientX - drag.x, deltaY = event.clientY - drag.y;if (Math.hypot(deltaX, deltaY) > 4) {camera.follow = false;camera.x = drag.cameraX - deltaX / camera.scale;camera.y = drag.cameraY + deltaY / camera.scale;render();}};
byId("map").onpointerup = (event) => {
  if (drag && Math.hypot(event.clientX - drag.x, event.clientY - drag.y) < 5) {
    const rect = byId("map").getBoundingClientRect();const pointX = event.clientX - rect.left, pointY = event.clientY - rect.top;
    const target = hitTargets.toReversed().find((item) => Math.hypot(item.x - pointX, item.y - pointY) < 25);
    if (target) selectFoot(target.id, target.pose, target.side); else {selectedId = null;byId("selection").hidden = true;render();}
  }
  drag = null;
};
byId("map").onpointercancel = () => {drag = null;};
byId("map").addEventListener("wheel", (event) => {event.preventDefault();zoom(event.deltaY < 0 ? 1.1 : 1 / 1.1);}, { passive: false });
// 页面隐藏时暂停，防止恢复显示后补算未观察到的控制时间
document.addEventListener("visibilitychange", () => {if (document.hidden) setRunning(false);});
new ResizeObserver(() => {drawMap();drawChart();drawCompass();}).observe(byId("map"));
// 将墙钟时间换算为整数控制拍，任何时刻最多保留一个在途推进请求
setInterval(() => {
  const now = performance.now(), delta = Math.min((now - lastTick) / 1000, .2);lastTick = now;
  if (!running || !current) return;
  accumulator = Math.min(.5, accumulator + delta * Number(byId("speed").value));
  if (advancePending) return;
  const frames = Math.min(25, Math.floor(accumulator / current.dt));
  if (frames < 1) return;
  accumulator -= frames * current.dt;
  advancePending = true;
  enqueue("/api/advance", { frames }).finally(() => {advancePending = false;});
}, 40);
reset();