# ECG Arrhythmia Classification - Advanced Denoising Guide

## 1. Overview & Documentation Reconciliation

There has been a notable discrepancy in earlier project documentation (e.g., `INCEPTIONTIME_PAPER1_STUDY_GUIDE.md` and `NSHT_PAPER3_STUDY_GUIDE.md`) which claimed the data pipeline explicitly *avoided* destructive filters to preserve QRS morphological fidelity. 

**The Reality:** The current structural backbone of the codebase relies on an **Advanced Denoising Pipeline** hardcoded directly into the dataset generation phase (`src/data/download.py`). Before any ECG segments are extracted and saved into the `.npy` raw data files, every signal is subjected to a heavy, 3-stage mathematical filtering process. 

This ensures that all downstream models (InceptionTime, EfficientNet, and NSHT) are actually operating on highly-smoothed, normalized signals rather than truly "raw" data.

---

## 2. The 3-Stage Denoising Pipeline (`download.py`)

During raw dataset extraction (specifically when running `python -m src.data.download --create-raw`), the targeted ECG leads undergo the following sequential sequence.

### Stage 1: Baseline Wander Removal (Highpass Filter)
Patients breathing and physical movement cause the electrical baseline to wander smoothly up and down.
* **Algorithm:** 2nd-Order Butterworth Highpass Filter
* **Cutoff Frequency:** 0.5 Hz
* **Implementation:** `scipy.signal.butter(2, 0.5 / nyq, btype='high')` followed by `filtfilt` for zero-phase distortion.
* **Purpose:** Forces the signal baseline flatly on the zero axis without temporally shifting the QRS peaks.

### Stage 2: Powerline Interference Removal (Notch Filter)
Electrical mains (A/C power) bleed a constant hum into medical equipment. Depending on the region, this is usually 50 Hz or 60 Hz.
* **Algorithm:** Infinite Impulse Response (IIR) Notch Filter
* **Target Frequency:** 60 Hz
* **Q-Factor:** 30 (controls the bandwidth of the notch)
* **Implementation:** `scipy.signal.iirnotch` followed by `filtfilt`.
* **Purpose:** Acts as a precise mathematical scalpel to remove exactly the 60 Hz sinusoidal noise without damaging the frequencies around it.

### Stage 3: High-Frequency Artifact Smoothing (DWT Soft-Thresholding)
Muscle contractions (EMG artifacts) and sensor static introduce high-frequency jitter. Simple lowpass filtering would round off the sharp R-peaks. Instead, Wavelet thresholding is used.
* **Algorithm:** Discrete Wavelet Transform (DWT)
* **Wavelet Type:** Daubechies 4 (`db4`)
* **Decomposition Level:** 5 Levels
* **Thresholding Logic:** Donoho-Johnstone Universal Threshold.
  1. The noise variance (`sigma`) is mathematically estimated using the Median Absolute Deviation (MAD) of the finest detail coefficients (`coeffs[-1]`), scaled by 0.6745.
  2. The universal limit (`uthresh`) is calculated as: `sigma * sqrt(2 * log(N))` where N is the signal length.
  3. **Soft Thresholding** is then applied to all detail coefficients, gently shrinking noise coefficients to zero while keeping structural wavelet coefficients intact.
* **Implementation:** Uses `pywt.wavedec`, `pywt.threshold(mode='soft')`, and `pywt.waverec`. 

---

## 3. Impact on Research Papers

Because this preprocessing happens *prior* to `X_raw.npy` generation, it intrinsically alters how the models behave relative to their theoretical whitepapers:

* **Paper 1 (InceptionTime)** & **Paper 2 (EfficientNet):** Both models heavily benefit from this pipeline. By mathematically removing noise beforehand, the models don't have to waste parameter capacity learning to ignore 60Hz hums or wandering baselines. They can focus entirely on morphological classification.
* **Paper 3 (NSHT Dual-Evolution):** The architecture for NSHT introduces a "Learnable Morlet wavelet front-end" explicitly designed to perform *adaptive denoising* during the forward pass. However, because `download.py` already forcefully denoises the data, the NSHT's learnable wavelets are currently performing *secondary* or *residual* denoising on an already clean signal.

---

## 4. Alternative "Standalone" Denoising Modules

While `download.py` handles the official pipeline, the repository also contains standalone exploratory modules aimed at breaking signal-to-noise ratio limits:

* **`frtv_denoising.py` & `denoise_and_balance_rpeak.py` (DWT-FrTV):** Uses a deeper `db6` wavelet with Fractional Total Variation and an advanced optimization loop (SALSA algorithm). It aggressively protects sharp edges (like QRS spikes) while heavily blurring background noise.
* **`48_87_dB_Denoiser.py` (Deep Neural Network Approach):** A Keras/TensorFlow model trained with a custom `ultra_snr_loss` function. It penalizes the 1st and 2nd derivatives of the signal to force an extremely smooth output curve, aiming for a >40 dB Signal-to-Noise Ratio.

---
*Note: If future experiments require truly "raw" data to test the NSHT's native adaptive denoising limits, the `ADVANCED DENOISING PIPELINE` lines in `download.py` must be disabled or conditionally vaulted behind a CLI flag.*