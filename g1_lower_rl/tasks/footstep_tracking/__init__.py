"""脚步跟踪训练组件，任务由 g1_lower_rl.tasks 统一注册"""

FOOTSTEP_PROFILES = ("tracking", "walk-first", "precision", "step-episode")


def make_footstep_env_cfg(profile: str, play: bool = False, *, precision_stage: dict[str, int] | None = None):
	"""统一选择训练阶段；保留旧profile名称和precision_stage恢复契约。"""
	if profile not in FOOTSTEP_PROFILES:
		raise ValueError(f"Unknown footstep profile: {profile}")
	if precision_stage and profile != "precision":
		raise ValueError("Landing ramp settings require the precision profile")
	if profile == "precision":
		from g1_lower_rl.tasks.footstep_tracking.walking import precision_env_cfg

		return precision_env_cfg(play=play, **(precision_stage or {}))
	if profile == "walk-first":
		from g1_lower_rl.tasks.footstep_tracking.walking import walk_first_env_cfg

		return walk_first_env_cfg(play=play)
	if profile == "step-episode":
		from g1_lower_rl.tasks.footstep_tracking.step_episode import step_episode_env_cfg

		return step_episode_env_cfg(play=play)
	from g1_lower_rl.tasks.footstep_tracking.env_cfg import footstep_env_cfg

	return footstep_env_cfg(play=play)
