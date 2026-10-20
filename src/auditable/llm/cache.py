"""A JSON prediction cache for LLM-based methods.

LLM calls cost money and are not reproducible, so a method can record its predictions
once and replay them. The cache file holds ``{"provenance": {...}, "predictions": {key:
prediction}}``. A key is the SHA-256 of the method, the strategy version, the model label
and the exact input the prompt is built from, so a changed trace or a changed prompt
version never reuses a stale answer.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Optional


def input_key(method_id: str, version: str, model: str, payload: Any) -> str:
    """The cache key for one input: SHA-256 over a canonical JSON of the four parts."""
    blob = json.dumps({"method": method_id, "version": version, "model": model, "input": payload},
                      sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "sha256-" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


class PredictionCache:
    """Predictions keyed by :func:`input_key`, optionally backed by a JSON file.

    With ``path`` set, an existing file is loaded and :meth:`save` writes atomically.
    """

    def __init__(self, path: Optional[str] = None, provenance: Optional[Dict[str, Any]] = None) -> None:
        self.path = path
        self.provenance: Dict[str, Any] = dict(provenance or {})
        self.predictions: Dict[str, Any] = {}
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            self.provenance = {**data.get("provenance", {}), **self.provenance}
            self.predictions = dict(data.get("predictions", {}))

    def get(self, key: str) -> Optional[Any]:
        return self.predictions.get(key)

    def put(self, key: str, prediction: Any) -> None:
        self.predictions[key] = prediction

    def __contains__(self, key: str) -> bool:
        return key in self.predictions

    def __len__(self) -> int:
        return len(self.predictions)

    def save(self, path: Optional[str] = None) -> str:
        path = path or self.path
        if not path:
            raise ValueError("no path to save the cache to")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"provenance": self.provenance, "predictions": self.predictions}, fh,
                      ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
        return path
