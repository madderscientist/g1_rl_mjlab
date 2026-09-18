"""平地脚步跟踪训练配置，观测和动作遵循独立 GRU 的部署契约"""

import copy

import torch
from mjlab.envs.mdp import generated_commands, last_action
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.noise import NoiseModelWithAdditiveBiasCfg

from g1_lower_rl.assets import WHOLE_BODY_JOINTS
from g1_lower_rl.tasks.footstep_tracking.commands import FootstepCommandCfg, footstep_execution_failed
from g1_lower_rl.tasks.footstep_tracking.rewards_cfg import make_rewards, make_terminations
from g1_lower_rl.tasks.lower_body.cfg.env_cfg import make_lower_body_env_cfg
from g1_lower_rl.tasks.lower_body.cfg.observations import make_observations as lower_body_observations


def joint_state(env, asset_cfg: SceneEntityCfg, velocity: bool = False) -> torch.Tensor:
  """按契约顺序读取绝对关节角或关节速度，默认角度只在模型内部扣除"""
  data = env.scene[asset_cfg.name].data
  return (data.joint_vel if velocity else data.joint_pos_biased)[:, asset_cfg.joint_ids]


def body_imu(env, asset_cfg: SceneEntityCfg, component: str = "both") -> torch.Tensor:
  """以刚体轴表示角速度和单位重力，不把世界角速度冒充 IMU 读数"""
  data = env.scene[asset_cfg.name].data
  orientation = data.body_link_quat_w[:, asset_cfg.body_ids].squeeze(1)
  angular = data.body_link_ang_vel_w[:, asset_cfg.body_ids].squeeze(1)
  gravity = torch.zeros_like(angular)
  gravity[:, 2] = -1
  if component == "angular":
    return quat_apply_inverse(orientation, angular)
  if component == "gravity":
    return quat_apply_inverse(orientation, gravity)
  if component != "both":
    raise ValueError(f"Unknown IMU component: {component}")
  return torch.cat((quat_apply_inverse(orientation, angular), quat_apply_inverse(orientation, gravity)), dim=-1)


def critic_state(env) -> torch.Tensor:
  """提供仅训练可见的线速度和接触，actor 不接收这些特权量"""
  return torch.cat((env.scene["robot"].data.root_link_lin_vel_b, env.scene["feet_ground_contact"].data.found.float()), dim=-1)


def footstep_env_cfg(play: bool = False):
  """复用基础 G1 平地场景，替换速度和高度任务的命令、观测、奖励与课程"""
  cfg = make_lower_body_env_cfg()
  cfg.scene.num_envs = 1 if play else 64
  cfg.commands = {"footsteps": FootstepCommandCfg(debug_vis=play)}
  joints = SceneEntityCfg("robot", joint_names=WHOLE_BODY_JOINTS, preserve_order=True)
  reference = lower_body_observations()["actor"].terms
  # 契约：29角度、29速度、两组IMU各6维、相位/频率及双脚支撑基准与两步目标共14维
  terms = {
    "joint_pos": ObservationTermCfg(
      func=joint_state, params={"asset_cfg": joints}, noise=copy.deepcopy(reference["joint_pos"].noise)
    ),
    "joint_vel": ObservationTermCfg(
      func=joint_state,
      params={"asset_cfg": copy.deepcopy(joints), "velocity": True},
      noise=copy.deepcopy(reference["joint_vel"].noise),
    ),
  }
  for prefix, body in (("pelvis", "pelvis"), ("torso", "torso_link")):
    for component, baseline in (("angular", "base_ang_vel"), ("gravity", "projected_gravity")):
      noise = copy.deepcopy(reference[baseline].noise)
      assert isinstance(noise, NoiseModelWithAdditiveBiasCfg)
      noise.bias_noise_cfg.operation = "abs"
      terms[f"{prefix}_{component}"] = ObservationTermCfg(
        func=body_imu,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=(body,)), "component": component},
        noise=noise,
      )
  terms["footsteps"] = ObservationTermCfg(func=generated_commands, params={"command_name": "footsteps"})
  for term in terms.values():
    term.delay_min_lag = 0
    term.delay_max_lag = 0
    term.delay_hold_prob = 0.0
  # 实体索引由观测管理器就地解析，actor/critic 配置不能共用可变对象
  critic = copy.deepcopy(terms)
  critic["privileged"] = ObservationTermCfg(func=critic_state)
  critic["actions"] = ObservationTermCfg(func=last_action)
  cfg.observations = {
    "actor": ObservationGroupCfg(terms=terms, concatenate_terms=True, enable_corruption=not play),
    "critic": ObservationGroupCfg(terms=critic, concatenate_terms=True, enable_corruption=False),
  }
  cfg.rewards = make_rewards(phase_cfg=cfg.commands["footsteps"].manager.phase)
  # mjlab 默认在奖励之后更新 command，首个终止项负责提前推进并冻结本拍奖励快照
  cfg.terminations = {"footstep_fault": TerminationTermCfg(func=footstep_execution_failed), **make_terminations()}
  cfg.curriculum = {}
  cfg.metrics = {}
  cfg.events.pop("gait_phase")
  cfg.events.pop("push_robot")
  cfg.episode_length_s = 60.0
  return cfg
