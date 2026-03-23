import json
import importlib
import random
from hashlib import sha1
from typing import Any, Dict, List, Optional

import numpy as np


def build_precompute_signature(
    *,
    context_length: int,
    lags_seq: List[int],
    prediction_length: int,
    filter_method: str,
    filter_config: Dict[str, Any],
    data_id_to_freq_map: Dict[int, str],
    max_windows_per_dataset: int,
    memory_cap_mb: float,
) -> str:
    """Builds a deterministic signature for strict in-run staleness checks."""
    payload = {
        "context_length": context_length,
        "lags_seq": list(lags_seq),
        "prediction_length": prediction_length,
        "filter_method": filter_method,
        "filter_config": filter_config,
        "data_id_to_freq_map": {int(k): str(v) for k, v in data_id_to_freq_map.items()},
        "max_windows_per_dataset": int(max_windows_per_dataset),
        "memory_cap_mb": float(memory_cap_mb),
    }
    return sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _get_filter_config(filter_processor) -> Dict[str, Any]:
    return {
        "dropout_rate": float(filter_processor.dropout_rate),
        "butter_cutoff": float(filter_processor.butter_cutoff),
        "butter_fs": float(filter_processor.butter_fs),
        "butter_order": int(filter_processor.butter_order),
        "base_period": (
            None
            if filter_processor.base_period is None
            else int(filter_processor.base_period)
        ),
        "h_order": float(filter_processor.h_order),
        "energy_threshold": float(filter_processor.energy_threshold),
    }


def _extract_data_id(instance: Dict[str, Any]) -> Optional[int]:
    value = instance.get("data_id")
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return int(value.reshape(-1)[0])
    return int(value)


def _instance_num_bytes(instance: Dict[str, Any]) -> int:
    total = 0
    for value in instance.values():
        if isinstance(value, np.ndarray):
            total += value.nbytes
    return total


def precompute_filtered_train_instances(
    *,
    data,
    estimator,
    module,
    filter_processor,
    data_id_to_freq_map: Dict[int, str],
    max_windows_per_dataset: int,
    memory_cap_mb: float,
    seed: int,
) -> Dict[str, Any]:
    """Precompute filtered train windows once and return in-memory cached instances."""
    try:
        cyclic_class = importlib.import_module("gluonts.itertools").Cyclic
    except ImportError:
        return {
            "enabled": False,
            "reason": "gluonts-not-available",
            "signature": None,
            "instances": None,
            "per_data_id_counts": {},
            "total_bytes": 0,
        }

    if filter_processor is None or filter_processor.method == "none":
        return {
            "enabled": False,
            "reason": "filtering-disabled",
            "signature": None,
            "instances": None,
            "per_data_id_counts": {},
            "total_bytes": 0,
        }

    num_train_datasets = len([k for k in data_id_to_freq_map.keys() if int(k) >= 0])
    num_train_datasets = max(1, num_train_datasets)
    total_windows_target = int(max_windows_per_dataset) * num_train_datasets
    if total_windows_target <= 0:
        return {
            "enabled": False,
            "reason": "zero-window-target",
            "signature": None,
            "instances": None,
            "per_data_id_counts": {},
            "total_bytes": 0,
        }

    signature = build_precompute_signature(
        context_length=int(estimator.context_length),
        lags_seq=list(estimator.lags_seq),
        prediction_length=int(estimator.prediction_length),
        filter_method=filter_processor.method,
        filter_config=_get_filter_config(filter_processor),
        data_id_to_freq_map=data_id_to_freq_map,
        max_windows_per_dataset=max_windows_per_dataset,
        memory_cap_mb=memory_cap_mb,
    )

    rng = random.Random(seed)
    stream = cyclic_class(data).stream()
    splitter = estimator._create_instance_splitter(module, "training")
    instances_iter = iter(splitter.apply(stream, is_train=True))

    memory_cap_bytes = int(memory_cap_mb * 1024 * 1024)
    total_bytes = 0
    instances: List[Dict[str, Any]] = []
    per_data_id_counts: Dict[int, int] = {}

    methods_requiring_freq = {"fits", "fits_then_cps"}

    for _ in range(total_windows_target):
        instance = next(instances_iter)

        data_id = _extract_data_id(instance)
        if data_id is None:
            return {
                "enabled": False,
                "reason": "missing-data-id-in-instance",
                "signature": signature,
                "instances": None,
                "per_data_id_counts": per_data_id_counts,
                "total_bytes": total_bytes,
            }

        freq = data_id_to_freq_map.get(int(data_id))
        if filter_processor.method in methods_requiring_freq and freq is None:
            return {
                "enabled": False,
                "reason": f"missing-frequency-for-data-id-{data_id}",
                "signature": signature,
                "instances": None,
                "per_data_id_counts": per_data_id_counts,
                "total_bytes": total_bytes,
            }

        past_target = np.asarray(instance["past_target"])
        filtered_target = filter_processor.process(
            past_target,
            freq=freq,
            data_id=int(data_id),
            context="precompute_train",
        )
        if isinstance(filtered_target, tuple):
            filtered_target = filtered_target[0]

        filtered_instance = dict(instance)
        filtered_instance["past_target"] = np.asarray(filtered_target, dtype=past_target.dtype)
        filtered_instance["is_prefiltered"] = np.asarray(1, dtype=np.int64)

        instance_bytes = _instance_num_bytes(filtered_instance)
        if memory_cap_bytes > 0 and total_bytes + instance_bytes > memory_cap_bytes:
            return {
                "enabled": False,
                "reason": "memory-cap-exceeded",
                "signature": signature,
                "instances": None,
                "per_data_id_counts": per_data_id_counts,
                "total_bytes": total_bytes,
            }

        instances.append(filtered_instance)
        total_bytes += instance_bytes
        per_data_id_counts[data_id] = per_data_id_counts.get(data_id, 0) + 1

    if len(instances) > 1:
        rng.shuffle(instances)

    return {
        "enabled": True,
        "reason": "ok",
        "signature": signature,
        "instances": instances,
        "per_data_id_counts": per_data_id_counts,
        "total_bytes": total_bytes,
    }
