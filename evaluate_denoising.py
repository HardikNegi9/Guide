"""
ECG Denoising Pipeline -- Standalone Evaluation & Comparison

Loads real PhysioNet ECG records (MIT-BIH, INCART, NSRDB), applies the exact
3-stage denoising pipeline used in src/data/download.py, and quantifies the
signal quality improvement at every stage.

Metrics
-------
* SNR  -- Signal-to-Noise Ratio (dB), treating the *fully denoised* signal
         as the reference "clean" ground truth.
* RMSE -- Root Mean Squared Error relative to clean.
* PRD  -- Percentage Root-mean-square Difference (clinical compression metric).

Outputs
-------
* Per-record table printed to stdout.
* ``denoising_evaluation/results.csv``  -- full numeric results.
* ``denoising_evaluation/comparison_<record>.png`` -- multi-panel plots for
  selected records showing raw vs each denoising stage.
* ``denoising_evaluation/snr_summary.png`` -- bar-chart of SNR boosts.

Usage
-----
    python scripts/evaluate_denoising.py                      # defaults
    python scripts/evaluate_denoising.py --db mitbih --max-records 5
    python scripts/evaluate_denoising.py --db all --max-records 3 --plot-records 3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")                      # non-interactive backend
import matplotlib.pyplot as plt
from scipy import signal as sp_signal

# ---------------------------------------------------------------------------
# Resolve project root so the script works from anywhere
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.download import DownloadConfig            # reuse paths & config


# ============================================================================
# METRICS
# ============================================================================

def compute_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """SNR in dB: 10*log10( sum(clean^2) / sum((clean - noisy)^2) )"""
    signal_power = np.sum(clean ** 2)
    noise_power = np.sum((clean - noisy) ** 2)
    if noise_power < 1e-30:
        return np.inf
    return 10.0 * np.log10(signal_power / noise_power)


def compute_rmse(clean: np.ndarray, estimate: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((clean - estimate) ** 2)))


def compute_prd(clean: np.ndarray, estimate: np.ndarray) -> float:
    """Percentage Root-mean-square Difference (clinical standard)."""
    num = np.sqrt(np.sum((clean - estimate) ** 2))
    den = np.sqrt(np.sum(clean ** 2))
    if den < 1e-30:
        return np.inf
    return float(100.0 * num / den)


# ============================================================================
# INDIVIDUAL DENOISING STAGES  (mirrors download.py exactly)
# ============================================================================

def stage_highpass(sig: np.ndarray, fs: int) -> np.ndarray:
    """Stage 1: Baseline wander removal -- 2nd-order Butterworth HP @ 0.5 Hz."""
    nyq = 0.5 * fs
    b, a = sp_signal.butter(2, 0.5 / nyq, btype="high")
    return sp_signal.filtfilt(b, a, sig)


def stage_notch(sig: np.ndarray, fs: int) -> np.ndarray:
    """Stage 2: Powerline interference removal -- IIR notch @ 60 Hz, Q=30."""
    nyq = 0.5 * fs
    b, a = sp_signal.iirnotch(60 / nyq, 30)
    return sp_signal.filtfilt(b, a, sig)


def stage_dwt(sig: np.ndarray) -> np.ndarray:
    """Stage 3: DWT soft-thresholding (db4, level 5, Donoho-Johnstone)."""
    import pywt
    coeffs = pywt.wavedec(sig, "db4", level=5)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2.0 * np.log(len(sig)))
    coeffs[1:] = [pywt.threshold(c, value=uthresh, mode="soft") for c in coeffs[1:]]
    return pywt.waverec(coeffs, "db4")


def full_pipeline(sig: np.ndarray, fs: int):
    """
    Run the complete 3-stage pipeline and return every intermediate result.

    Returns
    -------
    dict with keys:
        raw, after_hp, after_notch, after_dwt (=clean), z_normed
    """
    after_hp    = stage_highpass(sig, fs)
    after_notch = stage_notch(after_hp, fs)
    after_dwt   = stage_dwt(after_notch)

    # Z-score normalisation (as in download.py)
    z_normed = (after_dwt - np.mean(after_dwt)) / (np.std(after_dwt) + 1e-8)

    return {
        "raw":         sig,
        "after_hp":    after_hp,
        "after_notch": after_notch,
        "after_dwt":   after_dwt,
        "z_normed":    z_normed,
    }


# ============================================================================
# RECORD LOADING
# ============================================================================

def load_record_signal(data_dir: Path, rec_name: str) -> np.ndarray:
    """Read channel-0 (usually MLII) from a wfdb record."""
    import wfdb
    rec_path = str(data_dir / rec_name)
    record = wfdb.rdrecord(rec_path)
    return record.p_signal[:, 0].astype(np.float64)


def list_records(data_dir: Path, max_records: int | None = None):
    """List *.dat records in a directory."""
    recs = sorted({p.stem for p in data_dir.glob("*.dat")})
    if max_records is not None:
        recs = recs[:max_records]
    return recs


# ============================================================================
# PLOTTING
# ============================================================================

STAGE_LABELS = [
    ("raw",         "[1] Raw Signal"),
    ("after_hp",    "[2] After Highpass (0.5 Hz)"),
    ("after_notch", "[3] After Notch (60 Hz)"),
    ("after_dwt",   "[4] After DWT Denoising"),
    ("z_normed",    "[5] Z-Score Normalised"),
]

STAGE_COLORS = ["#8B8B8B", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def plot_comparison(stages: dict, fs: int, rec_label: str, out_path: Path,
                    window_sec: float = 5.0):
    """
    Multi-panel waveform comparison for a single record.

    Shows a ``window_sec``-second window of each pipeline stage side-by-side
    plus an overlay panel.
    """
    n_stages = len(STAGE_LABELS)
    window_samples = int(window_sec * fs)

    # Centre the window somewhere interesting (middle of the recording)
    mid = len(stages["raw"]) // 2
    lo = max(0, mid - window_samples // 2)
    hi = lo + window_samples
    t = np.arange(lo, hi) / fs

    fig, axes = plt.subplots(n_stages + 1, 1, figsize=(16, 3.2 * (n_stages + 1)),
                             sharex=True)
    fig.suptitle(f"Denoising Pipeline - {rec_label}  (fs = {fs} Hz)",
                 fontsize=16, fontweight="bold", y=0.995)

    for i, ((key, label), color) in enumerate(zip(STAGE_LABELS, STAGE_COLORS)):
        ax = axes[i]
        data = stages[key][lo:hi]
        ax.plot(t, data, color=color, linewidth=0.8, alpha=0.9)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=8)

    # Overlay panel: raw vs fully denoised
    ax_overlay = axes[-1]
    raw_seg  = stages["raw"][lo:hi]
    dwt_seg  = stages["after_dwt"][lo:hi]
    ax_overlay.plot(t, raw_seg, color="#8B8B8B", linewidth=0.6, alpha=0.55,
                    label="Raw")
    ax_overlay.plot(t, dwt_seg, color="#2ca02c", linewidth=1.0, alpha=0.9,
                    label="Denoised (DWT)")
    ax_overlay.set_ylabel("[6] Overlay", fontsize=9)
    ax_overlay.set_xlabel("Time (s)", fontsize=10)
    ax_overlay.legend(loc="upper right", fontsize=8)
    ax_overlay.grid(True, alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_snr_summary(results: list[dict], out_path: Path):
    """
    Grouped bar chart: SNR after each stage for every record.
    """
    records = [r["record"] for r in results]
    snr_hp    = [r["snr_after_hp"]    for r in results]
    snr_notch = [r["snr_after_notch"] for r in results]
    snr_raw   = [r["snr_raw"]         for r in results]

    x = np.arange(len(records))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(records) * 0.9), 6))
    bars1 = ax.bar(x - width, snr_raw,   width, label="Raw -> Clean SNR (dB)",
                   color="#8B8B8B", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x,         snr_hp,    width, label="After HP -> Clean SNR (dB)",
                   color="#1f77b4", edgecolor="white", linewidth=0.5)
    bars3 = ax.bar(x + width, snr_notch, width, label="After Notch -> Clean SNR (dB)",
                   color="#ff7f0e", edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Record", fontsize=11)
    ax.set_ylabel("SNR relative to fully-denoised signal (dB)", fontsize=11)
    ax.set_title("Denoising SNR Boost per Stage", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(records, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Value labels on bars
    for bars in (bars1, bars2, bars3):
        for bar in bars:
            h = bar.get_height()
            if np.isfinite(h):
                ax.annotate(f"{h:.1f}",
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=6)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_boost_summary(results: list[dict], out_path: Path):
    """
    Horizontal bar chart showing total SNR boost (raw -> fully denoised)
    for each record -- the single most important number.
    """
    records = [r["record"] for r in results]
    boosts  = [r["snr_boost_total"] for r in results]

    fig, ax = plt.subplots(figsize=(10, max(4, len(records) * 0.45)))
    colors  = plt.cm.viridis(np.linspace(0.25, 0.85, len(records)))

    y_pos = np.arange(len(records))
    bars = ax.barh(y_pos, boosts, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(records, fontsize=9)
    ax.set_xlabel("Total SNR Boost (dB)", fontsize=11)
    ax.set_title("Total Denoising SNR Improvement per Record", fontsize=14,
                 fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, boosts):
        if np.isfinite(val):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f} dB", va="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# MAIN EVALUATION LOOP
# ============================================================================

def evaluate_database(db_key: str, max_records: int | None,
                      plot_records: int, out_dir: Path) -> list[dict]:
    """Evaluate one database and return per-record metrics."""
    cfg = DownloadConfig.DATABASES[db_key]
    data_dir = cfg["local_dir"]
    fs = cfg["fs"]

    if not data_dir.exists():
        print(f"  [!] {cfg['name']} data not found at {data_dir} -- skipping.")
        return []

    records = list_records(data_dir, max_records)
    if not records:
        print(f"  [!] No records in {data_dir}.")
        return []

    print(f"\n{'='*70}")
    print(f"  {cfg['name']}  ({len(records)} records, fs={fs} Hz)")
    print(f"{'='*70}")

    results = []
    plotted = 0

    for rec_name in records:
        try:
            sig = load_record_signal(data_dir, rec_name)
        except Exception as e:
            print(f"  [X] Error loading {rec_name}: {e}")
            continue

        stages = full_pipeline(sig, fs)

        # Reference = fully denoised (after_dwt), as in the production pipeline.
        clean = stages["after_dwt"]

        # Trim to the same length (waverec can add ±1 sample)
        min_len = min(len(stages["raw"]), len(clean))
        raw_t    = stages["raw"][:min_len]
        hp_t     = stages["after_hp"][:min_len]
        notch_t  = stages["after_notch"][:min_len]
        clean_t  = clean[:min_len]

        snr_raw         = compute_snr(clean_t, raw_t)
        snr_after_hp    = compute_snr(clean_t, hp_t)
        snr_after_notch = compute_snr(clean_t, notch_t)

        rmse_raw        = compute_rmse(clean_t, raw_t)
        rmse_after_hp   = compute_rmse(clean_t, hp_t)

        prd_raw         = compute_prd(clean_t, raw_t)

        # The total boost is the SNR gap between raw and the next-to-clean
        # intermediate (after_notch -> clean is the DWT contribution).
        snr_boost_total = snr_after_notch - snr_raw  # total from HP + Notch
        snr_boost_dwt   = float("inf")  # DWT stage maps noisy -> clean by definition

        row = {
            "database":        cfg["name"],
            "record":          f"{db_key}_{rec_name}",
            "fs":              fs,
            "duration_sec":    len(sig) / fs,
            "snr_raw":         round(snr_raw, 2),
            "snr_after_hp":    round(snr_after_hp, 2),
            "snr_after_notch": round(snr_after_notch, 2),
            "snr_boost_hp":    round(snr_after_hp - snr_raw, 2),
            "snr_boost_notch": round(snr_after_notch - snr_after_hp, 2),
            "snr_boost_total": round(snr_after_notch - snr_raw, 2),
            "rmse_raw":        round(rmse_raw, 6),
            "rmse_after_hp":   round(rmse_after_hp, 6),
            "prd_raw_pct":     round(prd_raw, 2),
        }
        results.append(row)

        # Print one-liner
        print(f"  {rec_name:>12s}  |  SNR raw->clean {snr_raw:7.2f} dB  "
              f"|  after HP {snr_after_hp:7.2f} dB  "
              f"|  after Notch {snr_after_notch:7.2f} dB  "
              f"|  boost {row['snr_boost_total']:+.2f} dB")

        # Waveform comparison plot for selected records
        if plotted < plot_records:
            plot_path = out_dir / f"comparison_{db_key}_{rec_name}.png"
            plot_comparison(stages, fs, f"{cfg['name']} / {rec_name}", plot_path)
            print(f"           -> saved plot: {plot_path.name}")
            plotted += 1

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the ECG denoising pipeline with SNR/RMSE/PRD metrics."
    )
    parser.add_argument(
        "--db", type=str, default="all",
        choices=["all", "mitbih", "incart", "nsrdb"],
        help="Which database(s) to evaluate (default: all).",
    )
    parser.add_argument(
        "--max-records", type=int, default=None,
        help="Limit the number of records per database (default: all).",
    )
    parser.add_argument(
        "--plot-records", type=int, default=3,
        help="Number of per-record waveform comparison plots to save (default: 3).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for outputs (default: <project>/denoising_evaluation).",
    )
    args = parser.parse_args()

    # Output directory
    out_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "denoising_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  ECG Denoising Pipeline -- Standalone Evaluation")
    print("  Pipeline: Highpass(0.5Hz) -> Notch(60Hz) -> DWT(db4, L5)")
    print(f"  Output  : {out_dir}")
    print("=" * 70)

    # Select databases
    if args.db == "all":
        db_keys = ["mitbih", "incart", "nsrdb"]
    else:
        db_keys = [args.db]

    all_results: list[dict] = []
    for db_key in db_keys:
        rows = evaluate_database(db_key, args.max_records, args.plot_records, out_dir)
        all_results.extend(rows)

    if not all_results:
        print("\n[!] No records were evaluated. Make sure data is downloaded first:")
        print("   python -m src.data.download --download-all")
        return

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    import csv
    csv_path = out_dir / "results.csv"
    fieldnames = list(all_results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n[OK] Results CSV saved: {csv_path}")

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    snr_raws   = [r["snr_raw"]         for r in all_results if np.isfinite(r["snr_raw"])]
    snr_hps    = [r["snr_after_hp"]    for r in all_results if np.isfinite(r["snr_after_hp"])]
    snr_notchs = [r["snr_after_notch"] for r in all_results if np.isfinite(r["snr_after_notch"])]
    boosts     = [r["snr_boost_total"] for r in all_results if np.isfinite(r["snr_boost_total"])]

    print(f"\n{'='*70}")
    print("  SUMMARY  ({} records across {} database(s))".format(
        len(all_results), len(db_keys)))
    print(f"{'='*70}")
    if snr_raws:
        print(f"  Mean SNR (raw -> clean)            : {np.mean(snr_raws):8.2f} dB")
    if snr_hps:
        print(f"  Mean SNR (after HP -> clean)        : {np.mean(snr_hps):8.2f} dB")
    if snr_notchs:
        print(f"  Mean SNR (after Notch -> clean)     : {np.mean(snr_notchs):8.2f} dB")
    if boosts:
        print(f"  Mean Total SNR Boost (HP+Notch)    : {np.mean(boosts):+8.2f} dB")
        print(f"  Max  Total SNR Boost               : {np.max(boosts):+8.2f} dB")
        print(f"  Min  Total SNR Boost               : {np.min(boosts):+8.2f} dB")

    # ------------------------------------------------------------------
    # Summary plots
    # ------------------------------------------------------------------
    plot_snr_summary(all_results, out_dir / "snr_per_stage.png")
    print(f"  [OK] Per-stage SNR chart : {out_dir / 'snr_per_stage.png'}")

    plot_boost_summary(all_results, out_dir / "snr_boost_total.png")
    print(f"  [OK] Total boost chart   : {out_dir / 'snr_boost_total.png'}")

    print(f"\nAll outputs saved to: {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
