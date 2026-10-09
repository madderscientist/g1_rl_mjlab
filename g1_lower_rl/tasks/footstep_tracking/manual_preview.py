"""Viser单次前移控件和独立回放配置，不改变训练任务"""

from dataclasses import dataclass, fields, replace
import math

import mujoco
import torch
import viser
from mjlab.viewer import ViserPlayViewer

from g1_lower_rl.assets import ARM_JOINTS
from g1_lower_rl.footsteps.one_shot import OneShotFootstepManager
from g1_lower_rl.footsteps.tensor_manager import STARTING, WALKING, STOPPING, STANDING, SETTLING, FAULT
from g1_lower_rl.tasks.footstep_tracking.commands import FootstepCommand, FootstepCommandCfg, foot_poses


class ManualFootstepCommand(FootstepCommand):
  manager_type = OneShotFootstepManager

  def create_gui(self, name, server, get_env_idx, on_change=None, request_action=None):
    self.notice = "等待站稳"
    self.first_foot = server.gui.add_dropdown("首脚", options=("左脚", "右脚"), initial_value="左脚")
    self.distance_input = server.gui.add_number("前移 x (m)", initial_value=.10, min=0., max=.35, step=.01)
    self.frequency_input = server.gui.add_slider("步频 f (Hz)", min=.8, max=1.8, step=.05,
      initial_value=.8 if self.cfg.single_foot else 1.2)
    self.execute_button = server.gui.add_button("执行前移", icon=viser.Icon.PLAYER_PLAY)
    self.reset_button = server.gui.add_button("重置站姿", icon=viser.Icon.REFRESH)
    self.status_display = server.gui.add_html("")

    @self.execute_button.on_click
    def execute(event):
      request_action("CUSTOM", {"type": "one_shot_forward", "distance": self.distance_input.value,
        "frequency": self.frequency_input.value, "first_side": 0 if self.first_foot.value == "左脚" else 1})

    @self.reset_button.on_click
    def reset(event):
      request_action("RESET")

  def start_forward(self, distance, frequency, first_side):
    env = self._env
    if env.reset_buf.any():
      raise ValueError("本回合已结束，请重置站姿")
    sensor = env.scene[self.cfg.sensor_name].data
    if not (sensor.current_contact_time >= .1).all() or env.episode_length_buf[0] < 25:
      raise ValueError("等待双脚接触并站稳")
    if self.robot.data.root_link_lin_vel_w[0].norm() > .25:
      raise ValueError("身体尚未站稳")
    self.batch.start_forward(distance, frequency, first_side, close_stance=not self.cfg.single_foot)
    self.notice = "执行中"
    print(f"MANUAL_FORWARD distance={distance:.3f} frequency={frequency:.3f} first_side={first_side} single_foot={self.cfg.single_foot}", flush=True)

  def update_gui(self):
    if not hasattr(self, "status_display"):
      return
    mode = int(self.batch.state["mode"][0])
    names = {STARTING: "起步", WALKING: "前移", STOPPING: "收步", SETTLING: "确认接触", STANDING: "站定", FAULT: "失败"}
    actual = foot_poses(self.robot.data, self.site_ids)
    goal = self.batch.state["terminal_feet"]
    distances = (actual[..., :2] - goal[..., :2]).norm(dim=-1)[0].cpu().tolist()
    yaw_error = actual[..., 2] - goal[..., 2]
    yaw_error = torch.atan2(yaw_error.sin(), yaw_error.cos()).abs()[0].cpu().tolist()
    frequency = float(self.batch.state["frequency"][0])
    moving = mode in (STARTING, WALKING, STOPPING, SETTLING)
    ended = bool(self._env.reset_buf.any())
    for handle in (self.first_foot, self.distance_input, self.frequency_input):
      handle.disabled = moving or ended
    self.execute_button.disabled = moving or ended
    if ended:
      label = "失败，已暂停"
    elif mode == STANDING and self.batch.executions:
      label = "已站定" if max(distances) <= .05 and max(yaw_error) <= math.radians(10) else "已停，落点有偏差"
    else:
      label = names.get(mode, "等待")
    self.status_display.content = (
      f"<b>{label}</b><br>{self.notice}<br>实际 f: {frequency:.2f} Hz"
      f"<br>左脚误差: {distances[0]*100:.1f} cm / {math.degrees(yaw_error[0]):.1f}°"
      f"<br>右脚误差: {distances[1]*100:.1f} cm / {math.degrees(yaw_error[1]):.1f}°"
    )


@dataclass(kw_only=True)
class ManualFootstepCommandCfg(FootstepCommandCfg):
  single_foot: bool = False

  def build(self, env):
    return ManualFootstepCommand(self, env)


def configure_manual_preview(cfg, *, single_foot=False):
  """保留策略接口，使用手臂零位和并列站姿执行手动脚步"""
  original = cfg.commands["footsteps"]
  command = ManualFootstepCommandCfg(**{field.name: getattr(original, field.name) for field in fields(original)})
  command.single_foot = single_foot
  command.compile_backend = False
  command.manager = replace(command.manager, frequency_range=(.8, 1.8), initial_frequency=1.2)
  command.source = replace(command.source, initial_standing_probability=1., frequency_range=(.8, 1.8),
    initial_frequency=1.2, frequency_rate_range=(0., 0.), automatic_commands=False, automatic_restart=False)
  cfg.commands["footsteps"] = command
  cfg.curriculum = {}
  cfg.events["reset_arm_pose"].params.update(target_scale=0., ramp_duration_s=0.)
  for name in ("arm_pose_ramp", "arm_pose_drift", "arm_torque"):
    cfg.events.pop(name, None)
  robot = cfg.scene.entities["robot"]
  original_spec = robot.spec_fn

  def zero_arm_spec():
    spec = original_spec()
    for name in ARM_JOINTS:
      constraint = spec.add_equality(name=f"preview_hold_{name}", type=mujoco.mjtEq.mjEQ_JOINT,
        objtype=mujoco.mjtObj.mjOBJ_JOINT, name1=name, solref=[.005, 1.], solimp=[.99, .999, .001, .5, 2.])
      constraint.data[:] = 0.
    return spec

  robot.spec_fn = zero_arm_spec
  cfg.events["reset_base"].params.update(pose_range={}, velocity_range={})
  cfg.events["reset_robot_joints"].params.update(position_range=(0., 0.), velocity_range=(0., 0.))
  cfg.scene.num_envs = 1
  cfg.auto_reset = False
  cfg.episode_length_s = 3600.
  return cfg


class ManualStepViewer(ViserPlayViewer):
  def _handle_custom_action(self, action, payload):
    if isinstance(payload, dict) and payload.get("type") == "one_shot_forward":
      command = self.env.unwrapped.command_manager.get_term("footsteps")
      with torch.no_grad(), self._sim_lock:
        try:
          command.start_forward(payload["distance"], payload["frequency"], payload["first_side"])
          self.resume()
        except ValueError as error:
          command.notice = str(error)
        command.update_gui()
      return True
    return super()._handle_custom_action(action, payload)

  def _execute_step(self):
    succeeded = super()._execute_step()
    command = self.env.unwrapped.command_manager.get_term("footsteps")
    ended = bool(self.env.unwrapped.reset_buf.any())
    if ended:
      self.pause()
      command.notice = "回合终止，保留现场"
    if ended or self._step_count % 10 == 0:
      command.update_gui()
    return succeeded and not ended

  def reset_environment(self):
    super().reset_environment()
    command = self.env.unwrapped.command_manager.get_term("footsteps")
    command.batch.executions = 0
    command.notice = "等待站稳"
    command.update_gui()
    self.resume()