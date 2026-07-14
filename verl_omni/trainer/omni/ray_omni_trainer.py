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
"""Omni (thinker/talker) V1 trainer — a PPOTrainerSync subclass registered via ``@register_trainer("omni_sync")``."""

from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync


@register_trainer("omni_sync")
class OmniPPOTrainerSync(PPOTrainerSync):
    """``PPOTrainerSync`` subclass that wires tokenizer/processor from ``OmniModelConfig``."""

    def _init_tokenizer(self):
        """Assign tokenizer and processor from the model config."""
        model_config = self.config.actor_rollout_ref.model
        trust_remote_code = self.config.data.get("trust_remote_code", False)
        model_config.trust_remote_code = trust_remote_code

        self.tokenizer = model_config.tokenizer
        self.processor = model_config.processor
