# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Entrypoint for omni model RL training.

This is a thin wrapper around verl's V1 trainer (``verl.trainer.main_ppo``).
It sets omni-specific Hydra defaults and delegates the actual training
orchestration to verl's ``TaskRunnerV1`` / ``run_ppo`` framework.
"""

import hydra
from omegaconf import OmegaConf
from verl.trainer.main_ppo import TaskRunnerV1, run_ppo
from verl.utils.device import auto_set_device


@hydra.main(config_path="./config", config_name="omni_trainer", version_base=None)
def main(config):
    """Main entry point for omni model training.

    Delegates to verl's V1 trainer infrastructure. The omni config
    (``omni_trainer.yaml``) inherits from verl's ``ppo_trainer`` and adds
    omni-specific fields (``architecture``, ``model_stage``, etc.).

    The config's ``trainer.v1.trainer_mode`` must be set to ``omni_sync``
    (or another registered omni trainer variant) to use the omni-specific
    ``OmniPPOTrainerSync`` subclass that replaces monkey-patched
    ``hf_processor`` / ``hf_tokenizer`` with ``OmniModelBase`` adapter methods.
    """
    auto_set_device(config)
    OmegaConf.resolve(config)

    # Ensure V1 trainer is used (omni models only support V1).
    config.trainer.use_v1 = True

    # Ensure the omni trainer subclass is registered before verl looks it up.
    # The import fires ``@register_trainer("omni_sync")`` which populates
    # verl's ``TRAINER_REGISTRY``.
    import verl_omni.trainer.omni.ray_omni_trainer  # noqa: F401

    # Delegate to verl's V1 trainer infrastructure.
    # ``trainer.v1.trainer_mode: omni_sync`` selects OmniPPOTrainerSync.
    # ``agent_loop_manager_class`` selects OmniAgentLoopManagerTQ.
    run_ppo(config, task_runner_class=TaskRunnerV1)


if __name__ == "__main__":
    main()
