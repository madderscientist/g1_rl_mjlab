"""与机器人任务及网络版本无关的监督蒸馏组件。"""

from .models import ModelSpec as ModelSpec
from .trainer import DaggerConfig as DaggerConfig
from .trainer import run_distillation as run_distillation