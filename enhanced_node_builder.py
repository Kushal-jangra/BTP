"""Contextual and log-aware node features for grouped log events.

Each distinct event template becomes one node, in first-seen order. The builder
can be used independently of the paper-aligned GloVe baseline, for example by
passing its output as ``torch_geometric.data.Data(x=...)``.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


class EnhancedNodeBuilder:
    """Build fixed-width contextual, categorical, and statistical node features.

    Component values use deterministic feature hashing into ``component_dim``
    signed bins, so dimensions remain the same across graph groups without
    fitting a vocabulary on test data. Severity uses a fixed one-hot vocabulary.
    The scaler is fit independently within each graph, avoiding state leakage
    between train/test graphs. Set ``standardize=False`` to retain raw scales.
    """

    SEVERITIES = ("TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL", "UNKNOWN")
    _SEVERITY_RE = re.compile(r"\b(TRACE|DEBUG|INFO|WARN(?:ING)?|ERROR|FATAL|CRITICAL)\b", re.I)
    _COMPONENT_RE = re.compile(r"\[([A-Za-z][\w.$-]{1,80})\]")
    _NUMBER_RE = re.compile(r"(?<![\w.])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str = None,
        component_dim: int = 32,
        standardize: bool = True,
        batch_size: int = 32,
    ) -> None:
        if component_dim < 1:
            raise ValueError("component_dim must be positive")
        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.component_dim = component_dim
        self.standardize = standardize
        self.batch_size = batch_size
        self._embedding_cache: Dict[str, torch.Tensor] = {}
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "EnhancedNodeBuilder needs Hugging Face Transformers. Install it with "
                "`pip install 'transformers>=4.30,<5'`."
            ) from exc
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @staticmethod
    def _first(record: Mapping[str, Any], names: Sequence[str], default: Any = None) -> Any:
        for name in names:
            value = record.get(name)
            if value is not None and str(value).strip():
                return value
        return default

    @classmethod
    def _template(cls, record: Mapping[str, Any]) -> str:
        value = cls._first(record, ("EventTemplate", "event_template", "template", "event", "message", "Content", "content"), "")
        return str(value)

    @classmethod
    def _severity(cls, record: Mapping[str, Any], message: str) -> str:
        value = cls._first(record, ("Level", "level", "Severity", "severity", "LogLevel", "log_level"))
        if value is None:
            match = cls._SEVERITY_RE.search(message)
            value = match.group(1) if match else "UNKNOWN"
        value = str(value).upper()
        if value == "WARNING":
            value = "WARN"
        if value == "CRITICAL":
            value = "FATAL"
        return value if value in cls.SEVERITIES else "UNKNOWN"

    @classmethod
    def _component(cls, record: Mapping[str, Any], message: str) -> str:
        value = cls._first(record, ("Component", "component", "Module", "module", "Subsystem", "subsystem", "Service", "service"))
        subsystem = cls._first(record, ("Subsystem", "subsystem"))
        if value is not None and subsystem is not None and str(value).lower() != str(subsystem).lower():
            value = f"{value}::{subsystem}"
        if value is None:
            match = cls._COMPONENT_RE.search(message)
            value = match.group(1) if match else "UNKNOWN"
        return str(value).strip().lower() or "UNKNOWN"

    @classmethod
    def _parameters(cls, record: Mapping[str, Any], message: str) -> List[float]:
        raw = cls._first(record, ("ParameterList", "parameter_list", "Parameters", "parameters", "params"))
        def finite_clipped(values):
            return [float(np.clip(value, -1e12, 1e12)) for value in values if np.isfinite(value)]

        if isinstance(raw, Mapping):
            raw = list(raw.values())
        if isinstance(raw, (list, tuple, np.ndarray)):
            values: List[float] = []
            for item in raw:
                if isinstance(item, (int, float, np.number)):
                    value = float(item)
                    if np.isfinite(value):
                        values.append(value)
                elif isinstance(item, str):
                    values.extend(float(x) for x in cls._NUMBER_RE.findall(item))
            return finite_clipped(values)
        return finite_clipped(float(x) for x in cls._NUMBER_RE.findall(message))

    def extract_template_embedding(self, templates: List[str]) -> torch.Tensor:
        """Compute attention-mask mean-pooled contextual template vectors."""
        if not templates:
            return torch.empty((0, int(self.model.config.hidden_size)), dtype=torch.float32)
        missing = list(dict.fromkeys(t for t in templates if t not in self._embedding_cache))
        outputs = []
        with torch.no_grad():
            for start in range(0, len(missing), self.batch_size):
                batch = self.tokenizer(
                    missing[start:start + self.batch_size], padding=True,
                    truncation=True, max_length=256, return_tensors="pt",
                )
                batch = {key: value.to(self.device) for key, value in batch.items()}
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                outputs.append(torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().float())
        if outputs:
            computed = torch.cat(outputs, dim=0)
            self._embedding_cache.update(zip(missing, computed.unbind(0)))
        return torch.stack([self._embedding_cache[t] for t in templates])

    def _component_features(self, component: str) -> np.ndarray:
        # BLAKE2 makes category features stable across Python processes/runs.
        digest = hashlib.blake2b(component.encode("utf-8"), digest_size=16).digest()
        result = np.zeros(self.component_dim, dtype=np.float32)
        index = int.from_bytes(digest[:8], "little") % self.component_dim
        result[index] = 1.0
        return result

    def extract_node_statistics(
        self, log_group: List[Dict[str, Any]], unique_events: List[str]
    ) -> torch.Tensor:
        """Return severity, component, frequency, position, and parameter stats."""
        if not unique_events:
            return torch.empty((0, len(self.SEVERITIES) + self.component_dim + 7), dtype=torch.float32)
        event_to_index = {event: i for i, event in enumerate(unique_events)}
        positions: List[List[int]] = [[] for _ in unique_events]
        severities: List[List[str]] = [[] for _ in unique_events]
        components: List[List[str]] = [[] for _ in unique_events]
        parameters: List[List[float]] = [[] for _ in unique_events]
        for position, record in enumerate(log_group):
            event = self._template(record)
            if event not in event_to_index:
                continue
            idx = event_to_index[event]
            message = str(self._first(record, ("Content", "content", "message", "Message"), event))
            positions[idx].append(position)
            severities[idx].append(self._severity(record, message))
            components[idx].append(self._component(record, message))
            parameters[idx].extend(self._parameters(record, message))

        n_records = max(len(log_group), 1)
        rows = []
        for idx in range(len(unique_events)):
            pos = positions[idx]
            # Component mode is deterministic on ties and independent of insertion order.
            component_values = components[idx] or ["UNKNOWN"]
            component = min(set(component_values), key=lambda item: (-component_values.count(item), item))
            severity_values = severities[idx] or ["UNKNOWN"]
            severity = min(set(severity_values), key=lambda item: (-severity_values.count(item), item))
            one_hot = np.zeros(len(self.SEVERITIES), dtype=np.float32)
            one_hot[self.SEVERITIES.index(severity)] = 1.0
            nums = np.asarray(parameters[idx], dtype=np.float64)
            param_stats = (
                [float(nums.mean()), float(nums.std()), float(nums.min()), float(nums.max())]
                if nums.size else [0.0, 0.0, 0.0, 0.0]
            )
            # Signed log compression keeps large IDs/timestamps from dominating
            # the GNN when a graph has only one template and cannot be z-scored.
            param_stats = [
                float(np.sign(value) * np.log1p(abs(value))) for value in param_stats
            ]
            frequency = len(pos) / n_records
            first = pos[0] / n_records if pos else 0.0
            last = pos[-1] / n_records if pos else 0.0
            rows.append(np.concatenate((
                one_hot, self._component_features(component),
                np.asarray([frequency, first, last, *param_stats], dtype=np.float32),
            )))
        return torch.tensor(np.stack(rows), dtype=torch.float32)

    def build_node_features(self, parsed_log_group: List[Dict[str, Any]]) -> torch.Tensor:
        """Construct X in first-occurrence node order for one chronological group."""
        if not parsed_log_group:
            return torch.empty((0, 0), dtype=torch.float32)
        unique_events = list(dict.fromkeys(self._template(row) for row in parsed_log_group))
        semantic = self.extract_template_embedding(unique_events)
        statistics = self.extract_node_statistics(parsed_log_group, unique_events)
        features = torch.cat((semantic, statistics), dim=1)
        if self.standardize and features.size(0) > 1:
            # Constant columns become zero; semantic/categorical channels remain fixed-width.
            values = StandardScaler().fit_transform(features.numpy())
            features = torch.from_numpy(np.asarray(values, dtype=np.float32))
        return features.contiguous().to(dtype=torch.float32)
