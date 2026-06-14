from __future__ import annotations

import json
from pathlib import Path
from typing import List

from server.entities import Request


def load_requests(path: Path) -> List[Request]:
    requests: List[Request] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        requests.append(Request(**payload))
    return requests
