from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from node import Node

CTE_DURATION_S = 92e-6
CTE_FREQ_HZ = 250_000.0
high_fs = 5e9
adc_fs = 1_000_000.0
CTE_NUM_SAMPLES = int(round(CTE_DURATION_S * high_fs))
CTE_ADC_NUM_SAMPLES = int(round(CTE_DURATION_S * adc_fs))

CHANNEL_CENTER_FREQ_HZ = 2_402_000_000.0
CTE_LPF_CUTOFF_HZ = 500_000.0
C_LIGHT = 299_792_458.0

NOISE_OFF_SNR_DB = 120.0

_propagation_noise_std = 0.0
_rx_noise_std = 0.0
_sample_phase_noise_std = 0.0


def _noise_std_from_snr_db(snr_db: float) -> float:
    if not np.isfinite(snr_db) or snr_db >= NOISE_OFF_SNR_DB:
        return 0.0
    return float(10 ** (-snr_db / 20.0))


def set_simulation_noise(
    propagation_snr_db: float = NOISE_OFF_SNR_DB,
    rx_snr_db: float = NOISE_OFF_SNR_DB,
    sample_phase_snr_db: float = NOISE_OFF_SNR_DB,
) -> None:
    global _propagation_noise_std, _rx_noise_std, _sample_phase_noise_std
    _propagation_noise_std = _noise_std_from_snr_db(propagation_snr_db)
    _rx_noise_std = _noise_std_from_snr_db(rx_snr_db)
    _sample_phase_noise_std = _noise_std_from_snr_db(sample_phase_snr_db)


def _complex_awgn(n: int, std: float) -> np.ndarray:
    return (std / np.sqrt(2.0)) * (
        np.random.randn(n) + 1j * np.random.randn(n)
    )


def _lowpass_fft(
    samples: np.ndarray, sample_rate: float, cutoff_hz: float
) -> np.ndarray:
    spectrum = np.fft.fft(samples)
    freqs = np.fft.fftfreq(len(samples), d=1.0 / sample_rate)
    spectrum[np.abs(freqs) > cutoff_hz] = 0
    return np.fft.ifft(spectrum)


def path_phase(tx_node: Node, rx_node: Node) -> float:
    distance = tx_node.distance_to(rx_node)
    return (
        -2.0
        * math.pi
        * (CHANNEL_CENTER_FREQ_HZ + tx_node.frequency_offset)
        * distance
        / C_LIGHT
    )


def propagate(
    tx_node: Node, rx_node: Node, signal: np.ndarray
) -> np.ndarray:
    phase = path_phase(tx_node, rx_node)
    propagated = signal * np.exp(1j * phase)
    if _propagation_noise_std > 0.0:
        propagated = propagated + _complex_awgn(len(signal), _propagation_noise_std)
    return propagated


def _generate_cte_signal(tx_node: Node, t: float) -> np.ndarray:
    t_local = np.arange(CTE_NUM_SAMPLES, dtype=np.float64) / high_fs
    phase = 2.0 * math.pi * CTE_FREQ_HZ * t_local
    return np.exp(1j * phase)


def _upconvert_cte_to_ble(
    tx_node: Node, t: float, baseband: np.ndarray
) -> np.ndarray:
    t_abs = t + np.arange(len(baseband), dtype=np.float64) / high_fs
    lo_phase = tx_node.initial_phase + 2.0 * math.pi * (
        CHANNEL_CENTER_FREQ_HZ + tx_node.frequency_offset
    ) * t_abs
    return baseband * np.exp(1j * lo_phase)


def _downconvert_received_signal_phase(
    rx_node: Node, t: float, signal: np.ndarray
) -> np.ndarray:
    t_abs = t + np.arange(len(signal), dtype=np.float64) / high_fs
    lo_phase = rx_node.initial_phase + 2.0 * math.pi * (
        CHANNEL_CENTER_FREQ_HZ + rx_node.frequency_offset
    ) * t_abs
    mixed = signal * np.exp(-1j * lo_phase)
    if _rx_noise_std > 0.0:
        mixed = mixed + _complex_awgn(len(mixed), _rx_noise_std)
    baseband_phase = np.angle(mixed)
    decim = int(high_fs / adc_fs)
    sampled = baseband_phase[::decim]
    if _sample_phase_noise_std > 0.0:
        sampled = np.angle(
            np.exp(
                1j
                * (sampled + np.random.randn(len(sampled)) * _sample_phase_noise_std)
            )
        )
    return sampled[:CTE_ADC_NUM_SAMPLES]


def cte_receive_phase(tx_node: Node, rx_node: Node, t: float) -> np.ndarray:
    """1-to-1 CTE at time t: return RX downconverted phase samples."""
    bb = _generate_cte_signal(tx_node, t)
    rf = _upconvert_cte_to_ble(tx_node, t, bb)
    rf_rx = propagate(tx_node, rx_node, rf)
    return _downconvert_received_signal_phase(rx_node, t, rf_rx)


