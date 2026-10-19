"""普通脚步预览的自动游走和手动步频控件，不修改训练配置"""

from dataclasses import dataclass, fields, replace
import math

import torch
from mjlab.viewer import ViserPlayViewer

from g1_lower_rl.tasks.footstep_tracking.commands import (
  FootstepCommand, FootstepCommandCfg, TorsoHeadingCommand, TorsoHeadingCommandCfg,
)


class PreviewTorsoHeadingCommand(TorsoHeadingCommand):
  def __init__(self, cfg, env):
    self.automatic = True
    self.manual_theta_deg = 0.0
    super().__init__(cfg, env)

  def set_theta(self, automatic, theta_deg):
    limit_deg = round(math.degrees(self.angle_limit), 10)
    if not math.isfinite(theta_deg) or not -limit_deg <= theta_deg <= limit_deg:
      raise ValueError(f"Theta must be in [{-limit_deg}, {limit_deg}] degrees")
    self.automatic = automatic
    self.manual_theta_deg = theta_deg
    if automatic:
      self.time_left.zero_()
    else:
      self.target.fill_(math.radians(theta_deg))
      self.speed.fill_(self.cfg.speed_range[1])

  def _resample_command(self, env_ids):
    if self.automatic:
      super()._resample_command(env_ids)
    else:
      self.target[env_ids] = math.radians(self.manual_theta_deg)
      self.speed[env_ids] = self.cfg.speed_range[1]

  def create_gui(self, name, server, get_env_idx, on_change=None, request_action=None):
    self.get_env_idx = get_env_idx
    limit_deg = round(math.degrees(self.angle_limit), 10)
    self.automatic_input = server.gui.add_checkbox("自动 theta", initial_value=self.automatic)
    self.theta_input = server.gui.add_slider("目标 theta (°)", min=-limit_deg, max=limit_deg,
      step=1.0, initial_value=self.manual_theta_deg)
    self.theta_display = server.gui.add_html("")

    def request_theta():
      request_action("CUSTOM", {"type": "torso_heading", "automatic": self.automatic_input.value,
        "theta_deg": self.theta_input.value})

    @self.automatic_input.on_update
    def change_mode(event):
      request_theta()

    @self.theta_input.on_update
    def change_theta(event):
      self.automatic_input.value = False
      request_theta()

    self.update_gui()

  def update_gui(self):
    if not hasattr(self, "theta_display"):
      return
    env_idx = self.get_env_idx()
    current_deg = math.degrees(float(self.command[env_idx, 0]))
    target_deg = math.degrees(float(self.target[env_idx]))
    self.theta_display.content = f"目标 theta: {target_deg:.1f}°; 当前指令 theta: {current_deg:.1f}°"


@dataclass(kw_only=True)
class PreviewTorsoHeadingCommandCfg(TorsoHeadingCommandCfg):
  def build(self, env):
    return PreviewTorsoHeadingCommand(self, env)


def configure_theta_preview(cfg):
  if "torso_heading" in cfg.commands:
    original = cfg.commands["torso_heading"]
    command = PreviewTorsoHeadingCommandCfg(**{field.name: getattr(original, field.name) for field in fields(original)})
    command.compile_backend = False
    cfg.commands["torso_heading"] = command
  return cfg


class ThetaPlayViewer(ViserPlayViewer):
  def _handle_custom_action(self, action, payload):
    if isinstance(payload, dict) and payload.get("type") == "torso_heading":
      command = self.env.unwrapped.command_manager.get_term("torso_heading")
      with torch.no_grad(), self._sim_lock:
        command.set_theta(payload["automatic"], payload["theta_deg"])
        command.update_gui()
      print(f"PREVIEW_THETA automatic={payload['automatic']} target_deg={payload['theta_deg']:.1f}", flush=True)
      return True
    return super()._handle_custom_action(action, payload)

  def _update_theta_gui(self):
    if "torso_heading" in self.env.unwrapped.cfg.commands:
      self.env.unwrapped.command_manager.get_term("torso_heading").update_gui()

  def _execute_step(self):
    succeeded = super()._execute_step()
    if self._step_count % 10 == 0:
      self._update_theta_gui()
    return succeeded

  def reset_environment(self):
    super().reset_environment()
    self._update_theta_gui()


class FrequencyFootstepCommand(FootstepCommand):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.automatic_source = self.batch.source_cfg

  def set_frequency(self, automatic, frequency):
    lower, upper = self.automatic_source.frequency_range
    if not math.isfinite(frequency) or not lower <= frequency <= upper:
      raise ValueError(f"Frequency must be in [{lower}, {upper}] Hz")
    self.batch.source_cfg = (self.automatic_source if automatic else
      replace(self.automatic_source, frequency_range=(frequency, frequency), initial_frequency=frequency))
    self.batch.state["rate_remaining"].zero_()
    if not automatic:
      self.batch.state["request_frequency"].fill_(frequency)

  def create_gui(self, name, server, get_env_idx, on_change=None, request_action=None):
    self.get_env_idx = get_env_idx
    lower, upper = self.automatic_source.frequency_range
    initial = self.automatic_source.initial_frequency
    self.automatic_input = server.gui.add_checkbox("自动步频", initial_value=True)
    self.frequency_input = server.gui.add_slider("步频 f (Hz)", min=lower, max=upper, step=.05,
      initial_value=initial if initial is not None else (lower + upper) / 2)
    self.frequency_display = server.gui.add_html("")

    def request_frequency():
      request_action("CUSTOM", {"type": "footstep_frequency", "automatic": self.automatic_input.value,
        "frequency": self.frequency_input.value})

    @self.automatic_input.on_update
    def change_mode(event):
      request_frequency()

    @self.frequency_input.on_update
    def change_frequency(event):
      self.automatic_input.value = False
      request_frequency()

    self.update_gui()

  def update_gui(self):
    if not hasattr(self, "frequency_display"):
      return
    frequency = float(self.batch.state["frequency"][self.get_env_idx()])
    self.frequency_display.content = f"实际步频: {frequency:.2f} Hz"


@dataclass(kw_only=True)
class FrequencyFootstepCommandCfg(FootstepCommandCfg):
  def build(self, env):
    return FrequencyFootstepCommand(self, env)


def configure_frequency_preview(cfg):
  """仅预览禁用命令编译，允许GUI替换频率源参数"""
  original = cfg.commands["footsteps"]
  command = FrequencyFootstepCommandCfg(**{field.name: getattr(original, field.name) for field in fields(original)})
  command.compile_backend = False
  cfg.commands["footsteps"] = command
  return cfg


class FrequencyPlayViewer(ThetaPlayViewer):
  def _handle_custom_action(self, action, payload):
    if isinstance(payload, dict) and payload.get("type") == "footstep_frequency":
      command = self.env.unwrapped.command_manager.get_term("footsteps")
      with torch.no_grad(), self._sim_lock:
        command.set_frequency(payload["automatic"], payload["frequency"])
        command.update_gui()
      print(f"PREVIEW_FREQUENCY automatic={payload['automatic']} target={payload['frequency']:.2f}", flush=True)
      return True
    return super()._handle_custom_action(action, payload)

  def _execute_step(self):
    succeeded = super()._execute_step()
    if self._step_count % 10 == 0:
      self.env.unwrapped.command_manager.get_term("footsteps").update_gui()
    return succeeded

  def reset_environment(self):
    super().reset_environment()
    self.env.unwrapped.command_manager.get_term("footsteps").update_gui()