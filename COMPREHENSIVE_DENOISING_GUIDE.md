# Comprehensive ECG Denoising Guide

This document is the definitive guide to all forms of denoising implemented within the `ClassficationECG` repository. It reconciles the differences between the stated architecture of the research papers and the actual codebase logic, and it deeply inspects the experimental mathematical and deep learning modules designed to push past standard Signal-to-Noise Ratio (SNR) boundaries.

---

## Part 1: The Native Repository Pipeline (Production)

Despite previous architectural documentation (e.g., `INCEPTIONTIME_PAPER1_STUDY_GUIDE.md`) claiming that the pipeline relies solely on non-destructive data preparation, the reality of `src/data/download.py` tells a different story. 

Before any ECG data is segmented and passed into models like InceptionTime or EfficientNet, the raw stream passes through an **Advanced Denoising Pipeline**. This happens irreversibly during the native `.npy` file generation.

### 1.1 Step-by-Step Production Denoising (`download.py`)

1. **Baseline Wander Removal (Highpass Filter)**
   * **Mechanism:** A 2nd-order Butterworth Highpass filter (`0.5 Hz` cutoff).
   * **Purpose:** Removes extremely low-frequency noise such as patient respiration or muscle movement which causes the overall signal baseline to drift up and down.
   * **Code Logic:** Implemented using `scipy.signal.butter` and applied via `filtfilt` to guarantee zero-phase shift (so QRS peak timings are not pushed left or right).

2. **Powerline Interference Removal (Notch Filter)**
   * **Mechanism:** IIR (Infinite Impulse Response) Notch filter.
   * **Purpose:** Suppresses the `60 Hz` electrical hum introduced by standard power mains during the ECG acquisition.
   * **Code Logic:** `scipy.signal.iirnotch` tuned precisely to 60Hz with a Q-factor of 30.

3. **High-Frequency DWT Denoising (Soft-Thresholding)**
   * **Mechanism:** Discrete Wavelet Transform (DWT) using `db4` (Daubechies 4) wavelets spanning 5 decomposition levels. 
   * **Threshold Calculation:** Computes universal thresholding via the Donoho-Johnstone method (`sigma = Median(abs(details)) / 0.6745`). The final threshold becomes `sigma * sqrt(2 * log(N))`.
   * **Application:** "Soft" mode thresholding softly shrinks high-frequency coefficients to zero. This removes high-frequency jitter (like EMG muscle artifact) while preserving the sharper edges of R-peaks far better than a standard flat lowpass filter.

**Architectural Consequence:** Papers 1 (InceptionTime) and 2 (EfficientNet) unknowingly rely on this clean, pre-denoised signal. Paper 3 (NSHT), which advertises a *learnable* wavelet front-end designed to do this adaptively, is actually performing dual/secondary denoising on an already artificially clean signal.

---

## Part 2: The DWT-FrTV Mathematical Deep Dive (`frtv_denoising.py`)

For situations where standard `db4` soft-thresholding destroys too much high-frequency edge information (blunting QRS peaks), the repository contains an advanced mathematical module called **DWT-FrTV** (Discrete Wavelet Transform + Fractional Total Variation). 

### 2.1 The Concept of Fractional TV
Standard Total Variation (TV) denoising minimizes the first derivative of the signal, forcing it into a series of ugly "stair steps." **Fractional Total Variation (FrTV)** uses a fractional derivative (e.g., $\alpha = 0.8$), smoothing the signal without losing the natural curvature of the ECG waves.

### 2.2 Adaptive Alpha via GrÃ¼nwaldâ€“Letnikov
Instead of using a fixed fractional derivative, `get_adaptive_alpha()` calculates $\alpha$ dynamically.
* **Math:** It uses the GrÃ¼nwaldâ€“Letnikov fractional-order difference operator. It pre-computes gamma functions to create a weight matrix. 
* **Edge Protection:** If the code detects a sharp gradient (like the QRS complex), it forces $\alpha$ lower to *preserve* the edge. If it detects flat areas, it pushes $\alpha$ higher to deeply smooth background noise.

### 2.3 Optimization Loop (SALSA)
Because minimizing FrTV combined with Wavelet shrinkage isn't linearly solvable, the code uses **SALSA** (Split Augmented Lagrangian Shrinkage Algorithm):
1. Takes the ECG and decomposes it into `db6` wavelets (Level 4).
2. Uses **Bivariate Shrinkage** on the detail coefficients (which exploits the parent-child relationship in wavelet trees rather than just thresholding them blindly).
3. Applies a sequence of Douglas-Rachford proximal operators to simultaneously enforce the Wavelet sparsity and the FrTV smoothness penalties iteratively (100 passes).

---

## Part 3: Deep Custom Loss Denoising (`48_87_dB_Denoiser.py`)

The repository also includes a Keras/Tensorflow Deep Learning approach aiming to shatter standard benchmark metrics by achieving an unprecedented **> 40 dB SNR**. 

### 3.1 The `ultra_snr_loss` Function
Standard MSE (Mean Squared Error) training fails to reproduce smooth continuous waves. This script implements an aggressive custom loss function that stacks multiple penalty terms:
* **Term 1 (MSE):** `tf.reduce_mean(tf.square(y_true - y_pred))` ensures the basic amplitude matches.
* **Term 2 (1st Derivative Penalty):** Defines `diff1 = y_pred[num+1] - y_pred[num]` and penalizes the square of this. This acts as a mathematical "Tension Factor" pulling consecutive dots together.
* **Term 3 (2nd Derivative Penalty):** Defines `diff2 = diff1[num+1] - diff1[num]` and penalizes it. This acts as a mathematical "Stiffness Factor", making sure the *slope* of the line doesn't change rapidly (killing jitter).
* **Final Loss Equation:** `MSE + (0.01 * smoothness_loss) + (0.005 * smoothness_loss2)`.

### 3.2 Evaluation Metrics
The script constantly validates success against three strictly defined clinical standard formulas:
1. **SNR (Signal-to-Noise Ratio):** Calculated as $10 \cdot \log10(\sum (\text{clean}^2) / \sum (\text{denoised} - \text{clean})^2)$. Aims for $> 40\text{ dB}$.
2. **RMSE (Root Mean Square Error):** Measures the average deviation in microvolts.
3. **PRD (Percentage Root-mean-square Difference):** Used in compression standards to ensure clinical readability is preserved without phase distortion.

---

## Conclusion

If the objective is standard multi-classification, stick to the `download.py` pipeline (which acts identically to placing 3 heavy physical medical filters on the patient recording). If the objective is edge-preserving mathematical perfection, `frtv_denoising.py` is the state of the art. If the goal is pure numerical benchmark dominance (highest possible SNR output), the deep-learning smoothness loss from `48_87_dB_Denoiser.py` leads the repository.