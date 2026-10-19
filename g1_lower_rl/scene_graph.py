"""CUDA graphs for fixed-shape scene work, independent of task and RL algorithm."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from weakref import WeakKeyDictionary

import torch
import warp as wp
from mjlab.actuator.builtin_group import _TARGET_TENSOR_MAP
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scene import Scene
from mjlab.sensor import BuiltinSensor, ContactSensor

logger = logging.getLogger(__name__)


@dataclass
class SceneGraphStats:
  controls: bool = False
  contact_sensors: int = 0


_installed: WeakKeyDictionary[Scene, SceneGraphStats] = WeakKeyDictionary()


@dataclass
class _CapturedGraph:
  graph: torch.cuda.CUDAGraph
  operation: Callable[[], None]

  def replay(self) -> None:
    self.graph.replay()


@torch.inference_mode()
def _capture(operation: Callable[[], None], outputs: list[torch.Tensor], device: str) -> _CapturedGraph:
  # Warmup must not advance contact timers or overwrite live controls.
  saved = [value.clone() for value in outputs]
  stream = torch.cuda.Stream(device=device)
  stream.wait_stream(torch.cuda.current_stream(device))
  with torch.cuda.stream(stream):
    for _ in range(3):
      operation()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
      operation()
    for value, reference in zip(outputs, saved, strict=True):
      value.copy_(reference)
  torch.cuda.current_stream(device).wait_stream(stream)
  # CUDA Graph holds addresses, not references to external index/parameter tensors.
  return _CapturedGraph(graph, operation)


@torch.inference_mode()
def enable_scene_cuda_graphs(env: ManagerBasedRlEnv) -> SceneGraphStats:
  """Capture supported scene kernels; leave dynamic/custom components eager.

  Install after managers expand simulation fields and before the first reset.
  As with mjlab's physics graphs, scene topology and tensor storage must remain
  fixed afterward. In-place targets, gains, randomization and partial resets work.
  """
  stats = SceneGraphStats()
  if torch.device(env.device).type != "cuda":
    logger.info("Scene CUDA graphs disabled on %s", env.device)
    return stats
  if env.scene in _installed:
    raise ValueError("Scene CUDA graphs are already installed")

  entities = tuple(env.scene.entities.values())
  supported_controls = all(
    not entity._custom_actuators
    and not entity._builtin_group._delayed_groups
    and not entity._fused_actuator_group._groups
    for entity in entities
  )
  if supported_controls and any(entity.is_actuated for entity in entities):
    controls = wp.to_torch(env.sim.wp_data.ctrl)
    groups = []
    for entity in entities:
      for key, (target_ids, ctrl_ids) in entity._builtin_group._index_groups.items():
        groups.append((getattr(entity.data, _TARGET_TENSOR_MAP[key]), target_ids,
          entity.indexing.ctrl_ids[ctrl_ids]))

    def write_controls() -> None:
      # TorchArray.__setitem__ switches to Warp's stream, outside Torch capture.
      for targets, target_ids, ctrl_ids in groups:
        controls[:, ctrl_ids] = targets[:, target_ids]

    graph = _capture(write_controls, [controls], env.device)
    env.scene.write_data_to_sim = graph.replay
    stats.controls = True
  else:
    logger.warning("Scene control graph skipped: requires non-delayed builtin actuators")

  # Only the exact upstream sensor is audited here; subclasses may add host state.
  scene_sensors = tuple(env.scene.sensors.values())
  sensors = tuple(sensor for sensor in scene_sensors
    if type(sensor) is ContactSensor and (sensor.cfg.track_air_time or sensor.cfg.history_length > 0))
  if any(type(sensor) not in (ContactSensor, BuiltinSensor) for sensor in scene_sensors):
    logger.warning("Scene contact graph skipped: custom/non-contact sensors retain their original update order")
    sensors = ()
  outputs = []
  for sensor in sensors:
    if sensor._air_time_state is not None:
      outputs.extend(vars(sensor._air_time_state).values())
    if sensor._history_state is not None:
      outputs.extend(sensor._history_state.values())

  if sensors:
    def update_contacts() -> None:
      for sensor in sensors:
        if sensor._air_time_state is not None:
          sensor._update_air_time_tracking(env.physics_dt)
        if sensor._history_state is not None:
          data = sensor._extract_sensor_data()
          for name, history in sensor._history_state.items():
            value = getattr(data, name)
            if value is not None:
              # Upstream rebinds roll() outputs; replay needs stable history storage.
              history.copy_(history.roll(1, dims=2))
              history[:, :, 0] = value

    contact_graph = _capture(update_contacts, outputs, env.device)

    def update(dt: float) -> None:
      if dt != env.physics_dt:
        raise ValueError("Scene contact graph requires the captured physics_dt")
      for entity in entities:
        entity.update(dt)
      for sensor in scene_sensors:
        sensor._invalidate_cache()
      contact_graph.replay()

    env.scene.update = update
    stats.contact_sensors = len(sensors)

  _installed[env.scene] = stats
  print(f"[INFO] Scene CUDA graphs: controls={stats.controls}, contact_sensors={stats.contact_sensors}", flush=True)
  return stats
