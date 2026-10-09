"""普通脚步预览的自动游走和手动步频控件，不修改训练配置"""

from dataclasses import dataclass, fields, replace
import math

import torch
from mjlab.viewer import ViserPlayViewer

from g1_lower_rl.tasks.footstep_tracking.commands import FootstepCommand, FootstepCommandCfg


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


class FrequencyPlayViewer(ViserPlayViewer):
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