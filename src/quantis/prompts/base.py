from __future__ import annotations

import json
from typing import Any, Dict, List


class BasePrompt:
    name: str = ""
    description: str = ""
    system_prompt: str = ""

    def build_user_prompt(self, snapshot: Dict[str, Any]) -> str:
        """Build user message from snapshot data.

        Default implementation dumps the snapshot as JSON.
        Subclasses may override to customise formatting.
        """
        return json.dumps(snapshot, ensure_ascii=False, indent=2, default=str)

    def build(self, snapshot: Dict[str, Any]) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.build_user_prompt(snapshot)},
        ]
