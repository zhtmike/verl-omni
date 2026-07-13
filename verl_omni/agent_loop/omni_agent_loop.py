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
"""V1-compatible agent loop for omni models.

Provides ``OmniAgentLoopWorkerTQ`` and ``OmniAgentLoopManagerTQ`` as
drop-in replacements for verl's stock ``AgentLoopWorkerTQ`` and
``AgentLoopManagerTQ``.  The omni variants are selected via verl's
existing ``agent_loop_manager_class`` config mechanism — no monkey-
patching required.

In Phase 2, the worker class is identical to ``AgentLoopWorkerTQ`` in
behaviour: processor and tokenizer are loaded by ``OmniModelConfig`` and
inherited through the ``AgentLoopWorker`` base class (``self.model_config``
→ ``self.tokenizer`` / ``self.processor``).  The explicit subclass exists
for clarity and as a future hook point for multimodal pre-processing in
``_run_prompt``.
"""

import ray
from verl.trainer.ppo.v1.agent_loop_tq import (
    AgentLoopManagerTQ,
    AgentLoopWorkerTQ,
)


@ray.remote
class OmniAgentLoopWorkerTQ(AgentLoopWorkerTQ):
    """V1 agent loop worker for omni models.

    Functionally identical to ``AgentLoopWorkerTQ`` in Phase 2 — the
    omni-specific processor/tokenizer is loaded by ``OmniModelConfig``
    and inherited through ``AgentLoopWorker``'s ``self.model_config``.

    This subclass exists as a named hook point for future multimodal
    pre-processing (e.g. extracting and encoding audio/video/image data
    before the standard agent loop rollout in ``_run_prompt``).
    """


class OmniAgentLoopManagerTQ(AgentLoopManagerTQ):
    """V1 agent loop manager that creates ``OmniAgentLoopWorkerTQ`` workers.

    ``AgentLoopManagerTQ.__init__`` hardcodes ``self.agent_loop_workers_class
    = AgentLoopWorkerTQ``.  We override **after** ``super().__init__()`` to
    swap in our omni worker class.  Because ``_init_agent_loop_workers()``
    is called later by ``create()`` (not in ``__init__``), the override
    takes effect before the workers are instantiated.

    Selected via config::

        actor_rollout_ref:
          rollout:
            agent:
              agent_loop_manager_class: verl_omni.agent_loop.omni_agent_loop.OmniAgentLoopManagerTQ
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override after super().__init__() because AgentLoopManagerTQ
        # hardcodes its own worker class in __init__.
        self.agent_loop_workers_class = OmniAgentLoopWorkerTQ
