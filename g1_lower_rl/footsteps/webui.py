"""本地脚步规划预览服务，可用 python -m g1_lower_rl.footsteps.webui 启动"""

from __future__ import annotations

import argparse
import copy
import json
import math
import mimetypes
import secrets
import threading
import time
from collections import deque
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from g1_lower_rl.footstep_contract import FOOTSTEP_SLOTS, PREVIEW_FORMAT
from g1_lower_rl.footsteps import (
  FootstepCommand,
  FootstepManager,
  FootstepManagerCfg,
  FootstepSamplerCfg,
  GaitRequest,
  RandomCommandCfg,
  RandomCommandSource,
)
from g1_lower_rl.footsteps.config import DEFAULT_FOOT_WIDTH
from g1_lower_rl.footsteps.footprint_geometry import load_footprint_geometry

ASSETS = Path(__file__).with_name("web")


def number(value: object, name: str, lower: float, upper: float) -> float:
  """验证请求中的有限数值及闭区间，拒绝将布尔值当作数值"""
  if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
    raise ValueError(f"{name}: expected a finite number")
  if not lower <= value <= upper:
    raise ValueError(f"{name}: expected {lower} <= value <= {upper}")
  return float(value)


def integer(value: object, name: str, lower: int, upper: int) -> int:
  """在数值范围校验后要求输入为整数值，并转换成 int"""
  result = number(value, name, lower, upper)
  if not result.is_integer():
    raise ValueError(f"{name}: expected an integer")
  return int(result)


def boolean(value: object, name: str) -> bool:
  """验证请求字段确为布尔类型，不接受字符串或整数替代"""
  if not isinstance(value, bool):
    raise TypeError(f"{name}: expected a boolean")
  return value


class PreviewSession:
  """一个页面独享的规划器及有容量上限的曲线和操作记录"""

  def __init__(self, options: dict):
    """校验页面参数并创建纯规划会话，不启用实测接触确认"""
    if not isinstance(options, dict):
      raise TypeError("config must be an object")
    if "reference_frequency" in options:
      raise ValueError("reference_frequency was removed; reload the UI and set distance_max to the geometric limit")
    self.seed = integer(options.get("seed", 7), "seed", 0, 2**32 - 1)
    low = number(options.get("frequency_min", 0.8), "frequency_min", 0.6, 2.0)
    high = number(options.get("frequency_max", 1.8), "frequency_max", low, 2.0)
    initial = number(options.get("initial_frequency", 1 / 0.6), "initial_frequency", low, high)
    direction_noise = math.radians(number(options.get("direction_noise", 40), "direction_noise", 0, 90))
    yaw_noise = math.radians(number(options.get("yaw_noise", 30), "yaw_noise", 0, 60))
    sampler = FootstepSamplerCfg(
      distance_range=(
        number(options.get("distance_min", 0.05), "distance_min", 0.05, 0.8),
        number(options.get("distance_max", FootstepSamplerCfg().distance_range[1]), "distance_max", 0.05, 0.8),
      ),
      distance_mean=number(options.get("distance_mean", 0.25), "distance_mean", 0.01, 0.8),
      distance_std=number(options.get("distance_std", 0.10), "distance_std", 0.001, 0.5),
      min_width=number(options.get("min_width", FootstepSamplerCfg().min_width), "min_width", 0.08, 0.5),
      max_width=number(options.get("max_width", 0.36), "max_width", 0.08, 0.6),
      direction_noise=(-direction_noise, direction_noise),
      yaw_noise=(-yaw_noise, yaw_noise),
    )
    cfg = FootstepManagerCfg(
      sampler=sampler,
      frequency_range=(low, high),
      initial_frequency=initial,
      frequency_slew_rate=number(options.get("slew_rate", 0.2), "slew_rate", 0.01, 1.0),
      hold_width=number(options.get("hold_width", DEFAULT_FOOT_WIDTH), "hold_width", 0.08, 0.5),
      require_contact_confirmation=False,
    )
    source_cfg = RandomCommandCfg(
      frequency_range=(low, high),
      initial_frequency=initial,
      frequency_rate_range=(
        number(options.get("frequency_rate_min", -0.3), "frequency_rate_min", -2, 0),
        number(options.get("frequency_rate_max", 0.3), "frequency_rate_max", 0, 2),
      ),
      frequency_rate_interval_s=number(options.get("frequency_rate_interval_s", 2.0), "frequency_rate_interval_s", 0.02, 30),
      stop_probability=number(options.get("stop_probability", RandomCommandCfg().stop_probability), "stop_probability", 0, 1),
      direction_range=(
        math.radians(number(options.get("direction_min", -180), "direction_min", -180, 180)),
        math.radians(number(options.get("direction_max", 180), "direction_max", -180, 180)),
      ),
      foot_heading_range=(
        math.radians(number(options.get("heading_min", 0), "heading_min", -180, 180)),
        math.radians(number(options.get("heading_max", 0), "heading_max", -180, 180)),
      ),
      automatic_commands=False,
      automatic_restart=False,
    )
    self.manager = FootstepManager(cfg, self.seed)
    self.source = RandomCommandSource(source_cfg, self.seed)
    self.random_frequency = True
    self.foot_geometry = load_footprint_geometry()
    feet = [[0.0, cfg.hold_width / 2, 0.0], [0.0, -cfg.hold_width / 2, 0.0]]
    self.manager.reset(feet, self.source.reset(request=GaitRequest(frequency=initial, walking=False)))
    self.history: deque = deque(maxlen=9000)
    self.landings: deque = deque(maxlen=600)
    self.events: deque = deque(maxlen=200)
    self.touched = time.monotonic()
    self.record_event("reset")
    self.history.append(self.sample())

  def record_event(self, kind: str, **values) -> None:
    """按模拟时间记录一次控制输入或模式切换"""
    self.events.append({"time": self.manager.elapsed, "kind": kind, **values})

  def sample(self, command: FootstepCommand | None = None) -> dict:
    """将给定或当前命令转换为一帧可序列化的时序数据"""
    command = self.manager.command() if command is None else command
    return {
      "time": self.manager.elapsed,
      "frequency": command.frequency,
      "target": self.manager.request.frequency,
      "random_rate": self.source.rate if self.random_frequency and command.mode == "walking" else None,
      "phase": command.phase,
      "mode": command.mode,
      "contacts": command.required_contact.tolist(),
    }

  def state(self, samples: list | None = None) -> dict:
    """组装页面所需的脚印、几何、配置和历史数据快照"""
    command = self.manager.command()
    cfg = self.manager.cfg
    maximum = cfg.sampler.distance_range[1]
    support_ids = [next((landing["id"] for landing in reversed(self.landings) if landing["side"] == side), side)
             for side in (0, 1)]
    return {
      **self.sample(command),
      "footsteps": command.footsteps.tolist(),
      "footsteps_w": command.footsteps_w.tolist(),
      "future_ids": command.future_ids.tolist(),
      "future_sides": command.future_sides.tolist(),
      "goal_ids": support_ids + command.future_ids.tolist(),
      "goal_sides": [0, 1, 0, 1],
      "anchor_w": command.anchor_w.tolist(),
      "supports": self.manager.supports.tolist(),
      "foot_geometry": copy.deepcopy(self.foot_geometry),
      "landings": list(self.landings),
      "events": list(self.events)[-20:],
      "samples": samples or [],
      "direction": math.degrees(self.manager.request.movement_direction),
      "heading": math.degrees(self.manager.request.foot_heading),
      "automatic": self.source.automatic_commands,
      "random_frequency": self.random_frequency,
      "frequency_range": list(self.manager.cfg.frequency_range),
      "frequency_rate_range": list(self.source.cfg.frequency_rate_range),
      "frequency_rate_interval_s": self.source.cfg.frequency_rate_interval_s,
      "effective_distance_max": maximum,
      "effective_width_range": [cfg.sampler.min_width, cfg.sampler.max_width],
      "dt": self.manager.cfg.control_dt,
      "seed": self.seed,
    }

  def advance(self, frames: object) -> dict:
    """推进指定数量的固定控制拍，记录计划落地并返回最新页面状态"""
    count = integer(frames, "frames", 1, 100)
    samples = []
    for _ in range(count):
      before_mode = self.manager.mode
      pending = list(self.manager.queue)
      update = self.manager.advance()
      for side in update.landed_sides:
        landed = next(step for step in pending if step.side == side)
        self.landings.append({"time": self.manager.elapsed, "side": side,
                              "pose": landed.pose_w.tolist(), "id": int(landed.target_id)})
      if before_mode != update.command.mode:
        self.record_event(update.command.mode)
      request = self.source.advance(
        self.manager.cfg.control_dt,
        mode=update.command.mode,
        frequency=update.command.frequency,
        random_frequency=self.random_frequency,
      )
      self.manager.apply_request(request)
      sample = self.sample()
      samples.append(sample)
      self.history.append(sample)
    return self.state(samples)

  def control(self, data: dict) -> dict:
    """处理方向、频率、自动模式和停走指令，返回下一拍可用状态"""
    action = data.get("action")
    if action == "direction":
      direction = number(data.get("direction"), "direction", -180, 180)
      heading = number(data.get("heading"), "heading", -180, 180)
      request = replace(self.source.request, movement_direction=math.radians(direction), foot_heading=math.radians(heading))
      self.source.set_request(request)
      self.manager.apply_request(request)
      self.record_event(action, direction=direction, heading=heading)
    elif action == "frequency":
      value = data.get("value")
      target = None if value is None else number(value, "frequency", *self.manager.cfg.frequency_range)
      frequency = (
        target
        if target is not None
        else float(min(max(self.manager.frequency, self.source.cfg.frequency_range[0]), self.source.cfg.frequency_range[1]))
      )
      request = replace(self.source.request, frequency=frequency)
      self.source.set_request(request)
      self.manager.apply_request(request)
      self.random_frequency = target is None
      self.record_event(action, target=target)
    elif action == "automation":
      enabled = boolean(data.get("enabled"), "enabled")
      self.source.set_automation(enabled)
      self.record_event(action, enabled=enabled)
    elif action in ("stop", "start"):
      request = replace(self.source.request, walking=action == "start")
      self.source.set_request(request)
      self.manager.apply_request(request)
      self.record_event(action)
    else:
      raise ValueError("Unknown control action")
    return self.state()

  def export(self) -> dict:
    """导出配置与保留范围内的轨迹记录，不承诺还原被截断的历史"""
    return {
      "format": PREVIEW_FORMAT,
      "footstep_slots": list(FOOTSTEP_SLOTS),
      "planning_only": True,
      "seed": self.seed,
      "config": {"manager": asdict(self.manager.cfg), "source": asdict(self.source.cfg)},
      "source_state": {
        "request": asdict(self.source.request),
        "automatic_commands": self.source.automatic_commands,
        "automatic_restart": self.source.automatic_restart,
        "random_frequency": self.random_frequency,
      },
      "retention": {"samples": 9000, "landings": 600, "events": 200},
      "history": list(self.history),
      "landings": list(self.landings),
      "events": list(self.events),
      "state": self.state(),
    }


class PreviewServer(ThreadingHTTPServer):
  """通过会话标识隔离页面，用锁保护同一规划器的连续更新"""

  daemon_threads = True

  def __init__(self, address: tuple[str, int]):
    """绑定本地监听地址并初始化会话容器和更新锁"""
    super().__init__(address, PreviewHandler)
    self.sessions: dict[str, PreviewSession] = {}
    self.lock = threading.Lock()

  def dispatch(self, path: str, payload: dict) -> dict:
    """清理过期会话，再串行分发重置、推进、控制或导出请求"""
    with self.lock:
      now = time.monotonic()
      for key in list(self.sessions):
        if now - self.sessions[key].touched > 3600:
          del self.sessions[key]
      if path == "/api/reset":
        session = PreviewSession(payload.get("config", {}))
        old = payload.get("session")
        if isinstance(old, str):
          self.sessions.pop(old, None)
        if len(self.sessions) >= 32:
          raise ValueError("Too many preview sessions; close unused sessions and retry later")
        token = secrets.token_urlsafe(24)
        self.sessions[token] = session
        return {"session": token, **session.state(list(session.history))}
      token = payload.get("session")
      if not isinstance(token, str) or token not in self.sessions:
        raise ValueError("Preview session expired; reset the preview")
      session = self.sessions[token]
      session.touched = now
      if path == "/api/advance":
        result = session.advance(payload.get("frames", 1))
      elif path == "/api/control":
        result = session.control(payload)
      elif path == "/api/export":
        result = session.export()
      else:
        raise ValueError("Unknown API endpoint")
      return {"session": token, **result}


class PreviewHandler(BaseHTTPRequestHandler):
  """仅提供白名单静态资源和同源 JSON 预览接口"""

  server: PreviewServer

  def log_message(self, format: str, *args) -> None:
    """关闭高频预览请求的默认访问日志，避免刷屏"""

  def reply(self, status: int, body: bytes, content_type: str) -> None:
    """发送带缓存禁用与内容安全策略的 HTTP 响应"""
    self.send_response(status)
    self.send_header("Content-Type", content_type)
    self.send_header("Content-Length", str(len(body)))
    self.send_header("Cache-Control", "no-store")
    self.send_header("X-Content-Type-Options", "nosniff")
    self.send_header(
      "Content-Security-Policy",
      "default-src 'self'; style-src 'self'; script-src 'self'; object-src 'none'; frame-ancestors 'none'",
    )
    self.end_headers()
    self.wfile.write(body)

  def do_GET(self) -> None:
    """按白名单返回本地页面资源，不允许任意文件路径访问"""
    if urlsplit(self.path).path == "/favicon.ico":
      self.reply(204, b"", "image/x-icon")
      return
    filename = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}.get(urlsplit(self.path).path)
    if filename is None or not (ASSETS / filename).is_file():
      self.reply(404, b"Not found", "text/plain")
      return
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    self.reply(200, (ASSETS / filename).read_bytes(), f"{content_type}; charset=utf-8")

  def do_POST(self) -> None:
    """校验同源与请求大小，处理 JSON 请求并返回可读错误"""
    origin = self.headers.get("Origin")
    if origin and urlsplit(origin).netloc != self.headers.get("Host"):
      self.reply(403, b'{"error":"Cross-origin request denied"}', "application/json")
      return
    try:
      length = int(self.headers.get("Content-Length", "0"))
      if not 0 < length <= 16384 or self.headers.get_content_type() != "application/json":
        raise ValueError("Expected a JSON request no larger than 16 KB")
      payload = json.loads(self.rfile.read(length))
      if not isinstance(payload, dict):
        raise TypeError("Expected a JSON object")
      result = self.server.dispatch(urlsplit(self.path).path, payload)
      self.reply(200, json.dumps(result, allow_nan=False).encode(), "application/json")
    except (ValueError, TypeError, RuntimeError) as exc:
      self.reply(400, json.dumps({"error": str(exc)}).encode(), "application/json")


def main() -> None:
  """解析端口并在本机运行预览服务，收到中断时退出"""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--port", type=int, default=8765)
  args = parser.parse_args()
  with PreviewServer(("127.0.0.1", args.port)) as server:
    print(f"Footstep preview: http://127.0.0.1:{server.server_port} (planning only)", flush=True)
    try:
      server.serve_forever()
    except KeyboardInterrupt:
      pass


if __name__ == "__main__":
  main()
