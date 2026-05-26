from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline_config import CONFIG


STATE_NORMAL = "normal"
STATE_ELEVATED = "elevated"
STATE_STRONG = "strong"
STATE_ARTIFACT = "artifact"


def apply_cognitive_load_state_machine(
    features_df: pd.DataFrame,
    baseline_score_mean: float,
    baseline_score_std: float,
    score_col: str = "Cognitive_Load_Proxy_Score",
    artifact_col: str = "artifact",
) -> tuple[pd.DataFrame, dict]:
    """Apply raw thresholding, hysteresis, and refractory stabilization."""
    df = features_df.copy()
    std = float(baseline_score_std)
    if not np.isfinite(std) or std == 0:
        std = 1e-12

    hysteresis_config = CONFIG.get("hysteresis", {})
    refractory_config = CONFIG.get("refractory", {})
    load_config = hysteresis_config.get("cognitive_load", {})
    legacy_multipliers = CONFIG["cognitive_load_proxy"]["state_threshold_std_multipliers"]

    enter_elevated = float(load_config.get("enter_elevated_std", legacy_multipliers["elevated"]))
    enter_strong = float(load_config.get("enter_strong_std", legacy_multipliers["strongly_elevated"]))
    exit_elevated = float(load_config.get("exit_elevated_std", enter_elevated))
    exit_strong = float(load_config.get("exit_strong_std", enter_strong))

    thresholds = {
        "baseline_score_mean": float(baseline_score_mean),
        "baseline_score_std": std,
        "elevated_threshold": float(baseline_score_mean + enter_elevated * std),
        "strongly_elevated_threshold": float(baseline_score_mean + enter_strong * std),
        "enter_elevated_threshold": float(baseline_score_mean + enter_elevated * std),
        "exit_elevated_threshold": float(baseline_score_mean + exit_elevated * std),
        "enter_strong_threshold": float(baseline_score_mean + enter_strong * std),
        "exit_strong_threshold": float(baseline_score_mean + exit_strong * std),
    }

    raw_states = _raw_states(
        df,
        score_col=score_col,
        artifact_col=artifact_col,
        elevated_threshold=thresholds["enter_elevated_threshold"],
        strong_threshold=thresholds["enter_strong_threshold"],
    )
    df["cognitive_load_state_raw"] = raw_states

    if bool(hysteresis_config.get("enabled", False)):
        hysteresis_states = _apply_hysteresis(
            df,
            score_col=score_col,
            artifact_col=artifact_col,
            thresholds=thresholds,
        )
    else:
        hysteresis_states = raw_states
    df["cognitive_load_state_hysteresis"] = hysteresis_states

    if bool(refractory_config.get("enabled", False)):
        final_states = _apply_refractory(
            hysteresis_states,
            refractory_windows=int(refractory_config.get("state_change_refractory_windows", 0)),
        )
    else:
        final_states = hysteresis_states
    df["cognitive_load_state"] = final_states

    report = {
        **thresholds,
        "hysteresis_enabled": bool(hysteresis_config.get("enabled", False)),
        "refractory_enabled": bool(refractory_config.get("enabled", False)),
        "state_change_refractory_windows": int(refractory_config.get("state_change_refractory_windows", 0)),
    }
    return df, report


def _raw_states(
    df: pd.DataFrame,
    score_col: str,
    artifact_col: str,
    elevated_threshold: float,
    strong_threshold: float,
) -> pd.Series:
    states = pd.Series(STATE_NORMAL, index=df.index, dtype=object)
    states.loc[df[score_col] > elevated_threshold] = STATE_ELEVATED
    states.loc[df[score_col] > strong_threshold] = STATE_STRONG
    states.loc[df[score_col].isna()] = np.nan
    if artifact_col in df.columns:
        states.loc[df[artifact_col] == True] = STATE_ARTIFACT
    return states


def _apply_hysteresis(
    df: pd.DataFrame,
    score_col: str,
    artifact_col: str,
    thresholds: dict,
) -> pd.Series:
    states = []
    current = STATE_NORMAL

    for _, row in df.iterrows():
        score = row.get(score_col)
        if artifact_col in df.columns and bool(row.get(artifact_col, False)):
            states.append(STATE_ARTIFACT)
            continue
        if pd.isna(score):
            states.append(np.nan)
            continue

        if current == STATE_STRONG:
            if score < thresholds["exit_strong_threshold"]:
                current = STATE_ELEVATED if score >= thresholds["exit_elevated_threshold"] else STATE_NORMAL
        elif current == STATE_ELEVATED:
            if score >= thresholds["enter_strong_threshold"]:
                current = STATE_STRONG
            elif score < thresholds["exit_elevated_threshold"]:
                current = STATE_NORMAL
        else:
            if score >= thresholds["enter_strong_threshold"]:
                current = STATE_STRONG
            elif score >= thresholds["enter_elevated_threshold"]:
                current = STATE_ELEVATED

        states.append(current)

    return pd.Series(states, index=df.index, dtype=object)


def _apply_refractory(states: pd.Series, refractory_windows: int) -> pd.Series:
    if refractory_windows <= 0:
        return states.copy()

    final = []
    current = None
    lockout_remaining = 0

    for state in states:
        if pd.isna(state):
            final.append(state)
            continue
        if state == STATE_ARTIFACT:
            final.append(state)
            continue
        if current is None:
            current = state
            final.append(current)
            continue

        if lockout_remaining > 0 and state != current:
            final.append(current)
            lockout_remaining -= 1
            continue

        if state != current:
            current = state
            lockout_remaining = refractory_windows

        final.append(current)
        if lockout_remaining > 0:
            lockout_remaining -= 1

    return pd.Series(final, index=states.index, dtype=object)
