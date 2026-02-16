from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import pm4py
from pm4py.objects.conversion.log.variants import to_data_frame as factory
from pm4py.statistics.attributes.log.get import get_all_trace_attributes_from_log


def read_xes_file(directory_data: Path, xes_file_name: str) -> Tuple[List[str], pd.DataFrame, List[str]]:
    """Read an XES file and return states, dataframe, and trace attribute keys.

    Args:
        directory_data: Directory that contains the XES file.
        xes_file_name: File name (with or without .xes extension).

    Returns:
        A tuple containing:
        - list of unique activity names (states)
        - event log as a DataFrame
        - list of trace attribute keys
    """
    xes_path = Path(directory_data) / f"{Path(xes_file_name).stem}.xes"
    if not xes_path.exists():
        raise FileNotFoundError(f"Dataset '{xes_file_name}' not found.")

    try:
        event_logs = pm4py.read_xes(str(xes_path))
        print(f"Event-Log loaded successfully: {len(event_logs)} Events found.")
        event_log_df = factory.apply(event_logs)
        matrix_states = event_log_df["concept:name"].unique().tolist()
        trace_keys_list = list(get_all_trace_attributes_from_log(event_logs)) # type: ignore
    except FileNotFoundError:
        print(f"File not found: {xes_path}")
        raise
    except Exception as exc:
        print(f"Reading error: {exc}")
        raise

    return matrix_states, event_log_df, trace_keys_list # type: ignore

def get_trace_duration(group: pd.DataFrame) -> Tuple[bool, dict | None]:
    """Compute duration statistics for a single trace group.

    Args:
        group: DataFrame containing events for one case.

    Returns:
        Tuple (has_duration, stats_dict). If no timestamps exist, stats_dict is None.
    """
    timestamps = group["time:timestamp"].dropna().tolist()
    if not timestamps:
        case_id = group["case:id"].iloc[0] if "case:id" in group.columns else None
        print(f"No valid timestamps found in group {case_id if case_id is not None else 'no ID'}.")
        return False, None

    seconds = (max(timestamps) - min(timestamps)).total_seconds()
    minutes = seconds / 60
    hours = minutes / 60
    return True, {
        "seconds": seconds,
        "minutes": minutes,
        "hours": hours,
        "days": hours / 24,
    }