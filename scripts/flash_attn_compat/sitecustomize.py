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

"""Probe-only startup fixes for the FlashAttention compatibility shim."""

try:
    from transformers.utils import import_utils

    import_utils.PACKAGE_DISTRIBUTION_MAPPING.setdefault("flash_attn", ["flash-attn"])
except Exception:
    # Keep sitecustomize best-effort so unrelated Python commands still start.
    pass
