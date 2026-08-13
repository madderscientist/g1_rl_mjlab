"""在 CPU 上闭环重跑部署链路的仿真骨架。

用 mjlab 编译出来的**同一个** MuJoCo 模型（同样的执行器增益、armature、接触参数），
按部署侧 ``g1_lower_body_policy`` 的算法装配观测、跑 ONNX、算目标位置，然后
``mujoco.mj_step``。不走 warp，所以训练占着 GPU 的时候也能跑。

和训练的差别只有：没有域随机化、没有上肢扰动、地面是理想平面。所以它验证的是
“观测顺序 / 四元数约定 / 相位推进 / 动作缩放这条链路对不对”以及“策略大致什么水平”，
不是“策略在真机上有多稳”。

``scripts/check_deploy_policy.py`` 和 ``scripts/compare_video.py`` 都建立在这上面。
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import mujoco
import numpy as np
import onnxruntime
from mjlab.entity import Entity

from g1_lower_rl.assets import ARM_JOINTS, get_robot_cfg

CONTROL_DT = 0.02
DECIMATION = 4
GAIT_PERIOD = 0.6
FOOT_SITES = ("left_foot", "right_foot")
FALL_HEIGHT = 0.35
SWING_CLEARANCE = 0.03
"""足底 site 高过站立高度多少算摆动腿。站立时 site z ≈ 0.009。"""

# 部署侧能重建的观测项。实际用哪几项、以什么顺序，看 ONNX 元数据的 observation_names。
_OBS_TERMS = (
  "base_ang_vel",
  "projected_gravity",
  "command_twist",
  "command_height",
  "phase",
  "joint_pos",
  "joint_vel",
  "actions",
)


def build_model() -> mujoco.MjModel:
  """训练用的机器人 + 一块地面。"""
  spec = Entity(get_robot_cfg()).spec
  # 棋盘纹理：跟拍镜头下机器人总在画面中心，没有地面参考就分不出“走了”和“原地踏步”。
  spec.add_texture(
    name="grid",
    type=mujoco.mjtTexture.mjTEXTURE_2D,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
    rgb1=[0.45, 0.50, 0.55],
    rgb2=[0.30, 0.34, 0.38],
    width=300,
    height=300,
  )
  spec.add_material(name="grid", texrepeat=[4, 4], reflectance=0.1).textures[
    mujoco.mjtTextureRole.mjTEXROLE_RGB
  ] = "grid"
  spec.worldbody.add_geom(
    name="floor",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[0, 0, 0.05],
    condim=3,
    friction=[0.8, 0.005, 0.0001],
    material="grid",
  )
  spec.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1], diffuse=[0.6, 0.6, 0.6])
  # 默认离屏缓冲只有 640x480，对比视频要两格堆一起，得先把它撑大。
  spec.visual.global_.offwidth = 1280
  spec.visual.global_.offheight = 960
  model = spec.compile()
  # 必须覆盖成任务里的 MujocoCfg。MJCF 自带 0.002，照它跑的话 4 个物理步只有 8 ms，
  # 而步态相位仍按 20 ms 推进——策略会原地踏步却走不动，看着像退化其实是时序错了。
  model.opt.timestep = 0.005
  model.opt.iterations = 10
  model.opt.ls_iterations = 20
  model.opt.ccd_iterations = 50
  model.vis.headlight.ambient[:] = 0.6
  model.vis.headlight.diffuse[:] = 0.7
  return model


class Policy(NamedTuple):
  """ONNX 会话加从元数据解出来的部署契约。

  ``joint_names`` 是模型全部关节（包括策略不驱动的手臂、夹爪），``action_*`` 三项只覆盖
  策略驱动的那些，``obs_*`` 三项只覆盖观测里出现的那些。**三者长度可以互不相同**：
  站立任务观测全身 29 轴、只驱动下肢 15 轴，用动作关节去拼观测会短 14 维。
  """

  session: onnxruntime.InferenceSession
  joint_names: list[str]
  default_pos: np.ndarray
  action_joint_names: list[str]
  action_default_pos: np.ndarray
  action_scale: np.ndarray
  obs_pos_joint_names: list[str]
  obs_pos_default: np.ndarray
  obs_vel_joint_names: list[str]


def load_policy(path: str) -> Policy:
  session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
  meta = dict(session.get_modelmeta().custom_metadata_map)
  names = meta["joint_names"].split(",")
  defaults = np.array([float(v) for v in meta["default_joint_pos"].split(",")])
  scale = np.array([float(v) for v in meta["action_scale"].split(",")])
  if "action_joint_names" in meta:
    action_names = meta["action_joint_names"].split(",")
  else:
    # 早期导出的模型没写这一项。按前 n 个截断对下肢任务恰好对得上，换个任务就会错位。
    action_names = names[: len(scale)]
    print(f"[WARN] {path} 缺 action_joint_names，退回按前 {len(scale)} 个关节截断")
  slot = {name: i for i, name in enumerate(names)}

  def defaults_of(subset: list[str], what: str) -> np.ndarray:
    missing = [name for name in subset if name not in slot]
    if missing:
      raise ValueError(f"{what} {missing} 不在元数据的 joint_names 里")
    return np.array([defaults[slot[name]] for name in subset])

  def obs_subset(term: str) -> list[str]:
    key = f"obs_{term}_joint_names"
    if key in meta:
      return meta[key].split(",")
    # 早期导出没写这两项，那时观测关节恰好等于动作关节。
    print(f"[WARN] {path} 缺 {key}，退回按动作关节装配观测")
    return list(action_names)

  obs_pos_names = obs_subset("joint_pos")
  obs_vel_names = obs_subset("joint_vel")
  return Policy(
    session,
    names,
    defaults,
    action_names,
    defaults_of(action_names, "动作关节"),
    scale,
    obs_pos_names,
    defaults_of(obs_pos_names, "joint_pos 观测关节"),
    obs_vel_names,
  )


class Index:
  """策略关节在 qpos / qvel / ctrl 里的下标，以及测高用的足底 site。

  动作关节和观测关节分开索引：站立任务观测全身 29 轴、只驱动下肢 15 轴，混用会错位。
  """

  def __init__(self, model: mujoco.MjModel, policy: Policy):
    def joints_of(names: Sequence[str], what: str) -> list[int]:
      # mj_name2id 找不到返回 -1，直接当下标用会静默地索到最后一个关节。
      ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names
      ]
      unknown = [n for n, i in zip(names, ids) if i < 0]
      if unknown:
        raise ValueError(f"模型里没有这些{what}关节: {unknown}")
      return ids

    joint_names = policy.action_joint_names
    joints = joints_of(joint_names, "动作")
    actuators = [
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
      for name in joint_names
    ]
    unknown = [n for n, a in zip(joint_names, actuators) if a < 0]
    if unknown:
      raise ValueError(f"模型里没有这些关节的执行器: {unknown}")
    self.qpos = model.jnt_qposadr[joints]
    self.qvel = model.jnt_dofadr[joints]
    self.actuator = actuators
    self.obs_qpos = model.jnt_qposadr[joints_of(policy.obs_pos_joint_names, "观测")]
    self.obs_qvel = model.jnt_dofadr[joints_of(policy.obs_vel_joint_names, "观测")]
    self.obs_default_pos = policy.obs_pos_default
    self.arm_actuator = [
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
      for name in ARM_JOINTS
    ]
    self.foot_site = [
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in FOOT_SITES
    ]
    self.joint_range = model.jnt_range[joints]
    self.ctrl_range = model.actuator_ctrlrange[actuators]


def reset(model, data, names, defaults) -> None:
  mujoco.mj_resetData(model, data)
  data.qpos[:3] = [0.0, 0.0, 0.793]
  data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
  for i, name in enumerate(names):
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint >= 0:
      data.qpos[model.jnt_qposadr[joint]] = defaults[i]
  mujoco.mj_forward(model, data)


def set_arm_mode(model, mode: str) -> None:
  """手臂怎么处理。

  ``hold``：保留 <position> 执行器，目标写 0——对应部署侧 ``passive_targets``
  全 0（实机上还有 FPC 的重力前馈托着，这里没有，所以会比实机垂得低）。

  ``limp``：把手臂执行器的 kp/kd 全清零，不施任何力矩，手臂只靠重力和关节阻尼自然
  下垂。看视频用这个，同时也是一个更狠的工况：摆动的手臂就是下肢要抗的扰动。
  """
  if mode != "limp":
    return
  arm = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ARM_JOINTS
  ]
  model.actuator_gainprm[arm, 0] = 0.0
  model.actuator_biasprm[arm, 1] = 0.0
  model.actuator_biasprm[arm, 2] = 0.0


def rollout(model, data, session, index, default_pos, scale, command, seconds, video):
  """跑一个场景，返回 (逐拍记录, 是否摔倒)。

  观测按 ONNX 元数据里的 ``observation_names`` 逐项拼，顺序就是部署契约。别改回写死的
  拼接：GRU 版 actor 去掉了 ``actions`` 项，写死就只能对上一种网络。
  """
  n_joints = len(default_pos)
  last_action = np.zeros(n_joints)
  input_specs = {value.name: value for value in session.get_inputs()}
  output_names = [value.name for value in session.get_outputs()]
  meta = dict(session.get_modelmeta().custom_metadata_map)
  obs_names = meta["observation_names"].split(",")
  unknown_terms = [name for name in obs_names if name not in _OBS_TERMS]
  if unknown_terms:
    raise ValueError(
      f"观测项 {unknown_terms} 没有部署侧实现，补进 _OBS_TERMS。已支持：{sorted(_OBS_TERMS)}"
    )
  obs = np.zeros((1, int(input_specs["obs"].shape[1])), np.float32)
  state_pairs = (("h_in", "h_out"), ("c_in", "c_out"))
  state = {}
  for input_name, output_name in state_pairs:
    if input_name not in input_specs:
      continue
    if output_name not in output_names:
      raise ValueError(f"ONNX 有输入 {input_name}，但没有对应输出 {output_name}")
    shape = tuple(dim if isinstance(dim, int) else 1 for dim in input_specs[input_name].shape)
    state[input_name] = np.zeros(shape, dtype=np.float32)
  unknown_inputs = set(input_specs) - {"obs", *state}
  if unknown_inputs:
    raise ValueError(f"不支持的 ONNX 输入: {sorted(unknown_inputs)}")
  command = np.asarray(command)
  walking = float(np.linalg.norm(command[:3])) >= 0.1
  log: dict[str, list] = {
    "target": [],
    "vel_b": [],
    "yaw_rate": [],
    "height": [],
    "swing": [],
    "qpos": [],
  }
  rotation = np.zeros(9)
  for step in range(int(seconds / CONTROL_DT)):
    w, x, y, z = data.qpos[3:7]  # MuJoCo 是 (w, x, y, z)
    gravity = np.array(
      [
        -2.0 * (x * z - w * y),
        -2.0 * (y * z + w * x),
        -(1.0 - 2.0 * (x * x + y * y)),
      ]
    )
    if walking:
      angle = 2.0 * np.pi * ((step * CONTROL_DT) % GAIT_PERIOD) / GAIT_PERIOD
      phase = np.array([np.sin(angle), np.cos(angle)])
    else:
      phase = np.zeros(2)
    parts = {
      "base_ang_vel": data.sensordata[:3],  # imu_ang_vel，模型里排第一个传感器
      "projected_gravity": gravity,
      "command_twist": command[:3],
      "command_height": command[3:4],
      "phase": phase,
      "joint_pos": data.qpos[index.obs_qpos] - index.obs_default_pos,
      "joint_vel": data.qvel[index.obs_qvel],
      "actions": last_action,
    }
    obs[0] = np.concatenate([parts[name] for name in obs_names])
    outputs = session.run(None, {"obs": obs, **state})
    last_action = outputs[output_names.index("actions")].reshape(-1).astype(np.float64)
    for input_name, output_name in state_pairs:
      if input_name in state:
        state[input_name] = outputs[output_names.index(output_name)]
    target = default_pos + scale * last_action  # 不裁剪，和训练一致
    data.ctrl[index.actuator] = target

    mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
    foot_z = data.site_xpos[index.foot_site, 2]
    log["target"].append(target)
    log["vel_b"].append(rotation.reshape(3, 3).T @ data.qvel[:3])
    log["yaw_rate"].append(data.qvel[5])
    log["height"].append(data.qpos[2] - foot_z.min())
    # 逐脚统计：取 min 就成了“双脚同时离地”，正常步态永远是 0。
    log["swing"].append((foot_z > SWING_CLEARANCE).mean())
    # 留给离线渲染：存了轨迹就能用统一机位重放，不必在推理时决定镜头。
    log["qpos"].append(data.qpos.copy())

    for _ in range(DECIMATION):
      mujoco.mj_step(model, data)
    if video is not None and step % 2 == 0:  # 25 fps
      video["renderer"].update_scene(data, camera=video["camera"])
      video["images"].append(video["renderer"].render())
    if data.qpos[2] < FALL_HEIGHT:
      break
  return {k: np.asarray(v) for k, v in log.items()}, data.qpos[2] < FALL_HEIGHT
