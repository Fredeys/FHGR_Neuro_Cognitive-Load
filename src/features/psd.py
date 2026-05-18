import numpy as np
from scipy.signal import welch

from src.pipeline_config import CONFIG


def compute_psd(
    signal: np.ndarray,
    fs: float,
    nperseg_max: int = int(CONFIG["psd"]["nperseg_max"]),
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Welch power spectral density."""
    signal = np.asarray(signal, dtype=float)
    nperseg = min(nperseg_max, len(signal))
    return welch(signal, fs=fs, nperseg=nperseg, detrend="constant", scaling="density")


def compute_bandpower(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> float:
    """Integrate PSD inside a frequency band."""
    low, high = band
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))


def compute_spectral_entropy(psd: np.ndarray, eps: float = 1e-12) -> float:
    """Compute normalized spectral entropy."""
    psd = np.asarray(psd, dtype=float)
    total = float(np.sum(psd))
    if total <= eps:
        return 0.0

    probabilities = psd / total
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log2(probabilities))
    normalizer = np.log2(len(probabilities)) if len(probabilities) > 1 else 1.0
    return float(entropy / normalizer)
