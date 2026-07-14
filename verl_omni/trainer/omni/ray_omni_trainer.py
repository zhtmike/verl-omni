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
"""V1 trainer for omni models — extends verl's PPOTrainer with omni hooks.

The omni trainer subclass replaces the monkey-patched ``hf_processor`` /
``hf_tokenizer`` calls in ``PPOTrainer._init_tokenizer`` with
``OmniModelBase`` adapter methods. All V1 infrastructure (TransferQueue,
ReplayBuffer, lifecycle hooks) is inherited unchanged.
"""

from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils.fs import copy_to_local

from verl_omni.pipelines.model_base import OmniModelBase


@register_trainer("omni_sync")
class OmniPPOTrainerSync(PPOTrainerSync):
    """V1 sync trainer with omni model adapter hooks.

    Differences from stock ``PPOTrainerSync``:
    - Uses ``OmniModelBase`` to configure processor/tokenizer (replaces
      monkey-patched ``hf_processor`` / ``hf_tokenizer``).
    - All V1 infrastructure (TransferQueue, ReplayBuffer, lifecycle hooks)
      is inherited unchanged.

    Selected via config::

        trainer:
          v1:
            trainer_mode: omni_sync

    Other omni trainer variants (e.g. ``omni_colocate_async``) can be
    added later by following the same pattern and inheriting from the
    corresponding verl trainer variant.
    """

    def _init_tokenizer(self):
        """Override to use OmniModelBase for processor/tokenizer config.

        Replaces the stock ``PPOTrainer._init_tokenizer`` which calls
        ``verl.utils.tokenizer.hf_processor`` and ``hf_tokenizer``
        directly.  For omni models, processor and tokenizer configuration
        is model-specific (e.g. Qwen3-Omni needs ``thinker_config`` RoPE
        helpers and a ``chat_template.json``-based template).
        ``OmniModelBase`` adapters handle this in a type-safe way.
        """
        model_config = self.config.actor_rollout_ref.model
        local_path = copy_to_local(model_config.path, use_shm=model_config.get("use_shm", False))
        trust_remote_code = self.config.data.get("trust_remote_code", False)
        model_config.trust_remote_code = trust_remote_code

        # Load the omni model adapter registered for (architecture, model_stage).
        self.omni_adapter = OmniModelBase.get_class(model_config)

        # Use adapter methods instead of monkey-patched hf_processor / hf_tokenizer.
        # These set ``self.tokenizer`` and ``self.processor`` on the trainer, which
        # are then consumed by ``_init_dataloader`` in the parent's ``_setup()``.
        self.tokenizer = self.omni_adapter.configure_tokenizer(local_path, model_config)
        self.processor = self.omni_adapter.configure_processor(local_path, model_config)
