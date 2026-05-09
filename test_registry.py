from verl_omni.workers.config import DiffusionModelConfig
from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.workers.config.diffusion.rollout import DiffusionRolloutAlgoConfig
import verl_omni.pipelines # triggers imports

model_cfg = DiffusionModelConfig(architecture="QwenImagePipeline")
model_cfg.algo = DiffusionRolloutAlgoConfig(algo_type="mix_grpo")
cls = DiffusionModelBase.get_class(model_cfg)
print(cls.__name__)

model_cfg.algo = DiffusionRolloutAlgoConfig(algo_type="flow_grpo")
cls = DiffusionModelBase.get_class(model_cfg)
print(cls.__name__)
