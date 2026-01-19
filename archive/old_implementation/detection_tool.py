#!/usr/bin/env python3
"""
Splash-Only Dive Detection System

Adds modular signal aggregation, CSV export, GUI-based frame tagging,
and evaluation (precision, recall, F1) for splash detection.
"""
import os
import cv2
import time
import csv
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import savgol_filter

# Optional matplotlib import for GUI functionality
try:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("⚠️  matplotlib not available - GUI tagging functionality will be disabled")
    MATPLOTLIB_AVAILABLE = False

# Optional sklearn import for evaluation metrics
try:
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️  sklearn not available - evaluation functionality will be limited")
    SKLEARN_AVAILABLE = False

# --- Existing splash detectors imported or defined elsewhere ---
try:
    from slAIcer import (
        detect_splash_motion_intensity,
        detect_splash_frame_diff,
        detect_splash_optical_flow,
        detect_splash_contours,
        detect_splash_combined,
        get_video_fps
    )
    SLACER_AVAILABLE = True
except ImportError as e:
    print(f"❌ Failed to import splash detectors: {e}")
    SLACER_AVAILABLE = False

    # Define fallback functions
    def detect_splash_motion_intensity(*args, **kwargs):
        return None, 0.0
    def detect_splash_frame_diff(*args, **kwargs):
        return None, 0.0
    def detect_splash_optical_flow(*args, **kwargs):
        return None, 0.0
    def detect_splash_contours(*args, **kwargs):
        return None, 0.0
    def detect_splash_combined(*args, **kwargs):
        return None, 0.0

# --- New: Signal Aggregator ---
class SplashSignalAggregator:
    """
    Aggregates multiple per-frame detection signals into a DataFrame.
    """
    def __init__(self, detector, methods=None, fusion_weights=None, min_thresholds=None,
                 temporal_consistency=False, min_splash_duration=3, min_detectors_required=1,
                 temporal_bonus_factor=0.8):
        self.detector = detector
        # methods: list of (name, func) tuples for signal extraction
        self.methods = methods or [
            ('motion', detect_splash_motion_intensity),
            ('diff', detect_splash_frame_diff),
            ('flow', detect_splash_optical_flow),
            ('contour', detect_splash_contours),
        ]

        # Weighted fusion based on per-method F1 scores from analysis
        if fusion_weights is None:
            # Default weights based on F1-scores from analysis results
            weights = {
                'flow':    0.884,
                'contour': 0.878,
                'diff':    0.872,
                'motion':  0.837
            }
            # Normalize weights to sum=1
            total = sum(weights.values())
            self.fusion_weights = {k: v/total for k, v in weights.items()}
        else:
            self.fusion_weights = fusion_weights

        # Per-method minimum thresholds
        self.min_thresholds = min_thresholds or {
            'flow': 0.05,      # Optical flow minimum
            'contour': 100.0,  # Contour area minimum (further relaxed)
            'diff': 150.0,     # Frame difference minimum (further relaxed)
            'motion': 3.0      # Motion intensity minimum
        }

        # Enhanced gating parameters
        self.min_detectors_required = min_detectors_required  # Require 1+ detectors by default (very permissive)

        # Temporal consistency settings with soft gating
        self.temporal_consistency = temporal_consistency
        self.min_splash_duration = min_splash_duration
        self.temporal_bonus_factor = temporal_bonus_factor  # Softer bonus for more permissive gating

        print(f"🎯 Fusion weights: {self.fusion_weights}")
        print(f"🚧 Minimum thresholds: {self.min_thresholds}")
        print(f"🔒 Minimum detectors required: {self.min_detectors_required}")
        if temporal_consistency:
            print(f"⏰ Temporal consistency: min duration {min_splash_duration} frames")
            print(f"🎚️  Temporal bonus factor: {temporal_bonus_factor} (balanced gating)")

    def process_video(self, video_path, config):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # Default to 30 FPS if not available
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        prev_gray = None
        records = []

        for idx in tqdm(range(total), desc="Aggregating signals"):
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Extract splash zone (bounding box) same as in main detector
            band_top = int(config.splash_zone_top_norm * h)
            band_bottom = int(config.splash_zone_bottom_norm * h)
            band_left = int(config.splash_zone_left_norm * w)
            band_right = int(config.splash_zone_right_norm * w)

            # Extract the splash detection region
            band = gray[band_top:band_bottom, band_left:band_right]

            prev_band = None
            if prev_gray is not None:
                prev_band = prev_gray[band_top:band_bottom, band_left:band_right]

            row = {'frame': idx}
            for name, func in self.methods:
                try:
                    if prev_band is None:
                        score = 0.0
                    else:
                        _, score = func(band, prev_band)
                    row[f'{name}_score'] = score
                except Exception:
                    row[f'{name}_score'] = np.nan
            records.append(row)
            prev_gray = gray
        cap.release()

        df = pd.DataFrame(records)

        # Calculate weighted combined score
        df['combined_score'] = self._calculate_weighted_combined_score(df)

        # Apply per-method minimum requirements (count of methods above threshold)
        df['min_detectors_count'] = self._count_detectors_above_threshold(df)

        # Boolean gating: require minimum number of detectors
        df['gate_pass'] = (df['min_detectors_count'] >= self.min_detectors_required)

        # Apply temporal consistency if enabled (soft gating)
        if self.temporal_consistency:
            df['temporal_bonus'] = self._apply_soft_temporal_consistency(df)
            # Final score uses progressive penalty based on detector count
            df['final_score'] = df['combined_score'] * df['temporal_bonus'] * self._calculate_gating_penalty(df)
        else:
            # Final score uses progressive penalty based on detector count
            df['final_score'] = df['combined_score'] * self._calculate_gating_penalty(df)

        # Add legacy columns for compatibility
        df['meets_minimums'] = df['gate_pass'].astype(int)
        if self.temporal_consistency:
            df['temporal_consistent'] = (df['temporal_bonus'] == 1.0).astype(int)

        return df

    def _calculate_weighted_combined_score(self, df):
        """Calculate weighted combination of detection scores"""
        combined = np.zeros(len(df))

        for name, _ in self.methods:
            score_col = f'{name}_score'
            if score_col in df.columns:
                weight = self.fusion_weights.get(name, 0.25)  # Default equal weight
                combined += weight * df[score_col].fillna(0)

        return combined

    def _count_detectors_above_threshold(self, df):
        """Count how many detection methods exceed their minimum thresholds"""
        detector_count = np.zeros(len(df))

        for name, _ in self.methods:
            score_col = f'{name}_score'
            if score_col in df.columns and name in self.min_thresholds:
                min_thresh = self.min_thresholds[name]
                detector_count += (df[score_col].fillna(0) >= min_thresh).astype(int)

        return detector_count

    def _calculate_gating_penalty(self, df):
        """Calculate progressive penalty based on number of detectors passing minimum thresholds"""
        penalties = np.ones(len(df))  # Start with no penalty

        # Progressive penalty based on detector count
        detector_counts = df['min_detectors_count']

        # No penalty for 2+ detectors (normal strong signals)
        penalties = np.where(detector_counts >= 2, 1.0, penalties)

        # Small penalty for exactly 1 detector (moderate signals)
        penalties = np.where(detector_counts == 1, 0.7, penalties)

        # Larger penalty for 0 detectors (weak signals)
        penalties = np.where(detector_counts == 0, 0.3, penalties)

        return penalties

    def _apply_soft_temporal_consistency(self, df):
        """Apply soft temporal consistency: bonus for frames near detected peaks with wider window"""
        from scipy import ndimage

        # Start with combined score above threshold (using 90th percentile)
        threshold = df['combined_score'].quantile(0.90)
        binary_detections = (df['combined_score'] > threshold).astype(int)

        # Label connected components (consecutive splash detections)
        labeled_array, num_features = ndimage.label(binary_detections)

        # Create output array with softer gating (higher baseline bonus)
        temporal_bonus = np.full(len(df), self.temporal_bonus_factor)  # Default to 0.8 baseline

        # Give full bonus to splash sequences that meet minimum duration and nearby frames
        for label in range(1, num_features + 1):
            mask = (labeled_array == label)
            duration = np.sum(mask)

            if duration >= self.min_splash_duration:
                # Find the boundaries of this splash sequence
                indices = np.where(mask)[0]
                start_idx = max(0, indices[0] - 2)  # Extend ±2 frames
                end_idx = min(len(df), indices[-1] + 3)  # +3 because range is exclusive

                # Give full bonus to sequence and nearby frames
                temporal_bonus[start_idx:end_idx] = 1.0

        return temporal_bonus

# --- Export signals to CSV ---
def export_signals_to_csv(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"📊 Signals exported to {output_path}")

# --- GUI-based Frame Tagger ---
def tag_frames_cli(video_path, output_csv, detector, config):
    if not MATPLOTLIB_AVAILABLE:
        print("❌ matplotlib not available - cannot use GUI tagging functionality")
        print("💡 Install matplotlib: pip install matplotlib")
        return

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Preload all frames in ROI for display
    frames = []
    for _ in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Extract splash zone (bounding box) same as in main detector
        band_top = int(config.splash_zone_top_norm * h)
        band_bottom = int(config.splash_zone_bottom_norm * h)
        band_left = int(config.splash_zone_left_norm * w)
        band_right = int(config.splash_zone_right_norm * w)

        # Extract the splash detection region
        band = gray[band_top:band_bottom, band_left:band_right]
        frames.append(band)
    cap.release()

    labels = np.zeros(len(frames), dtype=int)
    idx = 0

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    img_disp = ax.imshow(frames[0], cmap='gray')
    ax.set_title(f"Frame {idx}")

    # Slider for frame navigation
    axframe = plt.axes([0.2, 0.05, 0.6, 0.03])
    frame_slider = Slider(axframe, 'Frame', 0, len(frames)-1,
                          valinit=0, valstep=1)

    def update(val):
        nonlocal idx
        idx = int(frame_slider.val)
        img_disp.set_data(frames[idx])
        ax.set_title(f"Frame {idx} | Label: {labels[idx]}")
        fig.canvas.draw_idle()
    frame_slider.on_changed(update)

    def on_key(event):
        nonlocal idx
        if event.key == 'right':
            idx = min(idx+1, len(frames)-1)
            frame_slider.set_val(idx)
        elif event.key == 'left':
            idx = max(idx-1, 0)
            frame_slider.set_val(idx)
        elif event.key == 'l':
            labels[idx] ^= 1  # toggle
            ax.set_title(f"Frame {idx} | Label: {labels[idx]}")
            fig.canvas.draw_idle()
        elif event.key == 'q':
            plt.close(fig)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    # Save labels
    df_labels = pd.DataFrame({'frame': np.arange(len(frames)), 'splash': labels})
    df_labels.to_csv(output_csv, index=False)
    print(f"✅ Labels saved to {output_csv}")

# --- Enhanced Evaluation Functions ---
def evaluate_single_threshold(df, threshold_quantile=0.90, threshold_value=None, show_details=True, score_column='combined_score'):
    """Evaluate performance at a single threshold with improved gating logic"""
    if not SKLEARN_AVAILABLE:
        print("❌ sklearn not available - cannot perform evaluation")
        return None

    y_true = df['splash'].astype(int)

    # Use final_score if available (includes all enhancements), otherwise fall back to combined_score
    if 'final_score' in df.columns and score_column == 'combined_score':
        score_column = 'final_score'
        print(f"📊 Using enhanced final_score (includes weighting, gating, temporal consistency)")

    # Improved quantile calculation: only consider frames that pass the gate
    if 'gate_pass' in df.columns:
        gated_frames = df[df['gate_pass']]
        if len(gated_frames) == 0:
            print("❌ No frames pass the gate - cannot evaluate")
            return None

        if threshold_value is not None:
            cutoff = threshold_value
        else:
            # Calculate quantile from gated scores only
            cutoff = gated_frames['combined_score'].quantile(threshold_quantile)

        # Prediction requires both passing gate AND being above threshold
        y_pred = ((df['combined_score'] >= cutoff) & df['gate_pass']).astype(int)

        print(f"🚪 Gate filtering: {len(gated_frames)}/{len(df)} frames pass gate ({len(gated_frames)/len(df)*100:.1f}%)")
    else:
        # Fallback to original logic if no gate_pass column
        if threshold_value is not None:
            cutoff = threshold_value
        else:
            cutoff = df[score_column].quantile(threshold_quantile)
        y_pred = (df[score_column] > cutoff).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary'
    )

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    if show_details:
        print(f"📊 Threshold: {cutoff:.3f} (quantile: {threshold_quantile:.3f})")
        print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1-score: {f1:.3f}")
        print(f"True Neg: {tn} | False Pos: {fp} | False Neg: {fn} | True Pos: {tp}")
        print("Confusion Matrix:")
        print(f" [[{tn:4d} {fp:4d}]  ← No-splash")
        print(f"  [{fn:4d} {tp:4d}]]  ← Splash")

    return {
        'threshold_quantile': threshold_quantile,
        'threshold_value': cutoff,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
        'score_column': score_column
    }

def quantile_sweep(df, quantiles=None, auto_select=False):
    """Sweep across multiple quantiles to find optimal threshold"""
    if quantiles is None:
        quantiles = [0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99]

    print("🔍 Running quantile sweep...")
    print("Quantile | Threshold | Precision | Recall | F1-Score")
    print("-" * 55)

    results = []
    best_f1 = 0
    best_result = None

    for q in quantiles:
        result = evaluate_single_threshold(df, threshold_quantile=q, show_details=False)
        if result:
            results.append(result)
            print(f"  {q:5.2f}  |   {result['threshold_value']:6.3f}  |   {result['precision']:6.3f}  |  {result['recall']:6.3f}  |  {result['f1']:6.3f}")

            if auto_select and result['f1'] > best_f1:
                best_f1 = result['f1']
                best_result = result

    if auto_select and best_result:
        print("\n🎯 Auto-selected optimal threshold:")
        print(f"Best F1: {best_result['f1']:.3f} at quantile {best_result['threshold_quantile']:.3f} (threshold: {best_result['threshold_value']:.3f})")
        return best_result

    return results

def analyze_error_cases(df, threshold_quantile=0.90):
    """Analyze false positives and false negatives with improved gating logic"""
    # Calculate threshold from gated frames if available
    if 'gate_pass' in df.columns:
        gated_frames = df[df['gate_pass']]
        if len(gated_frames) == 0:
            print("❌ No frames pass the gate - cannot analyze errors")
            return pd.DataFrame(), pd.DataFrame()
        cutoff = gated_frames['combined_score'].quantile(threshold_quantile)
        # Predictions require both passing gate AND being above threshold
        predicted_splash = (df['combined_score'] >= cutoff) & df['gate_pass']
    else:
        # Fallback to final_score if available, otherwise combined_score
        score_column = 'final_score' if 'final_score' in df.columns else 'combined_score'
        cutoff = df[score_column].quantile(threshold_quantile)
        predicted_splash = df[score_column] > cutoff

    # Extract error cases
    df_fp = df[(df['splash'] == 0) & predicted_splash]  # False Positives
    df_fn = df[(df['splash'] == 1) & ~predicted_splash]  # False Negatives

    print(f"\n🔍 Enhanced Error Analysis (threshold: {cutoff:.3f})")
    if 'gate_pass' in df.columns:
        print(f"🚪 Gate filtering: {df['gate_pass'].sum()}/{len(df)} frames pass gate")
    print(f"False Positives: {len(df_fp)} frames")
    print(f"False Negatives: {len(df_fn)} frames")

    if len(df_fp) > 0:
        print("\n❌ False Positive Examples (predicted splash, actually no-splash):")
        if 'gate_pass' in df.columns:
            print("Frame | Final    | Combined | Gate | MinDet | Motion | Diff   | Flow   | Contour | TempBonus")
            print("-" * 95)
            for _, row in df_fp.head(5).iterrows():
                gate_pass = row.get('gate_pass', 'N/A')
                min_det = row.get('min_detectors_count', 'N/A')
                temp_bonus = row.get('temporal_bonus', 'N/A')
                final_score = row.get('final_score', 'N/A')
                print(f"{row['frame']:5.0f} | {final_score:8.3f} | {row['combined_score']:8.3f} | {gate_pass:4} | {min_det:6} | {row['motion_score']:6.1f} | {row['diff_score']:6.1f} | {row['flow_score']:6.3f} | {row['contour_score']:7.1f} | {temp_bonus:9.3f}")
        else:
            print("Frame | Combined | Motion | Diff   | Flow   | Contour")
            print("-" * 55)
            for _, row in df_fp.head(5).iterrows():
                print(f"{row['frame']:5.0f} | {row['combined_score']:8.3f} | {row['motion_score']:6.1f} | {row['diff_score']:6.1f} | {row['flow_score']:6.3f} | {row['contour_score']:7.1f}")

    if len(df_fn) > 0:
        print("\n❌ False Negative Examples (predicted no-splash, actually splash):")
        if 'gate_pass' in df.columns:
            print("Frame | Final    | Combined | Gate | MinDet | Motion | Diff   | Flow   | Contour | TempBonus")
            print("-" * 95)
            for _, row in df_fn.head(5).iterrows():
                gate_pass = row.get('gate_pass', 'N/A')
                min_det = row.get('min_detectors_count', 'N/A')
                temp_bonus = row.get('temporal_bonus', 'N/A')
                final_score = row.get('final_score', 'N/A')
                print(f"{row['frame']:5.0f} | {final_score:8.3f} | {row['combined_score']:8.3f} | {gate_pass:4} | {min_det:6} | {row['motion_score']:6.1f} | {row['diff_score']:6.1f} | {row['flow_score']:6.3f} | {row['contour_score']:7.1f} | {temp_bonus:9.3f}")
        else:
            print("Frame | Combined | Motion | Diff   | Flow   | Contour")
            print("-" * 55)
            for _, row in df_fn.head(5).iterrows():
                print(f"{row['frame']:5.0f} | {row['combined_score']:8.3f} | {row['motion_score']:6.1f} | {row['diff_score']:6.1f} | {row['flow_score']:6.3f} | {row['contour_score']:7.1f}")

    return df_fp, df_fn

def analyze_per_method_performance(df, threshold_quantile=0.90):
    """Analyze performance of individual detection methods"""
    print(f"\n📊 Per-Method Analysis")
    methods = ['motion_score', 'diff_score', 'flow_score', 'contour_score']

    print("Method   | Threshold | Precision | Recall | F1-Score")
    print("-" * 50)

    for method in methods:
        if method in df.columns:
            method_cutoff = df[method].quantile(threshold_quantile)
            y_true = df['splash'].astype(int)
            y_pred = (df[method] > method_cutoff).astype(int)

            if SKLEARN_AVAILABLE:
                precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
                method_name = method.replace('_score', '').capitalize()
                print(f"{method_name:8s} | {method_cutoff:9.3f} | {precision:9.3f} | {recall:6.3f} | {f1:8.3f}")

def evaluate_signals(signals_csv, labels_csv, threshold_quantile=0.90,
                    sweep_quantiles=False, auto_threshold=False,
                    analyze_errors=False, per_method=False, plot_scores=False):
    """Enhanced evaluation with multiple analysis options"""
    if not SKLEARN_AVAILABLE:
        print("❌ sklearn not available - cannot perform evaluation")
        print("💡 Install sklearn: pip install scikit-learn")
        return

    df_signals = pd.read_csv(signals_csv)
    df_labels = pd.read_csv(labels_csv)
    df = pd.merge(df_signals, df_labels, on='frame')

    # Run quantile sweep if requested
    if sweep_quantiles:
        if auto_threshold:
            best_result = quantile_sweep(df, auto_select=True)
            if best_result:
                threshold_quantile = best_result['threshold_quantile']
        else:
            quantile_sweep(df)
            return

    # Main evaluation
    print("\n" + "="*60)
    print("🎯 MAIN EVALUATION")
    print("="*60)
    result = evaluate_single_threshold(df, threshold_quantile)

    # Additional analyses
    if per_method:
        analyze_per_method_performance(df, threshold_quantile)

    if analyze_errors:
        analyze_error_cases(df, threshold_quantile)

    if plot_scores and MATPLOTLIB_AVAILABLE:
        plot_evaluation_results(df, threshold_quantile)

def plot_evaluation_results(df, threshold_quantile=0.90):
    """Plot detection scores over time with threshold and ground truth"""
    if not MATPLOTLIB_AVAILABLE:
        print("❌ matplotlib not available - cannot plot results")
        return

    cutoff = df['combined_score'].quantile(threshold_quantile)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot 1: Combined score over time
    ax1.plot(df['frame'], df['combined_score'], 'b-', alpha=0.7, label='Combined Score')
    ax1.axhline(y=cutoff, color='r', linestyle='--', label=f'Threshold ({cutoff:.3f})')

    # Mark ground truth splash frames
    splash_frames = df[df['splash'] == 1]
    ax1.scatter(splash_frames['frame'], splash_frames['combined_score'],
               c='red', s=20, alpha=0.8, label='True Splash', marker='o')

    ax1.set_ylabel('Combined Score')
    ax1.set_title('Detection Scores Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Individual method scores
    methods = ['motion_score', 'diff_score', 'flow_score', 'contour_score']
    colors = ['blue', 'green', 'orange', 'purple']

    for method, color in zip(methods, colors):
        if method in df.columns:
            # Normalize scores to 0-1 for comparison
            normalized = (df[method] - df[method].min()) / (df[method].max() - df[method].min())
            ax2.plot(df['frame'], normalized, color=color, alpha=0.6,
                    label=method.replace('_score', '').capitalize())

    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Normalized Scores')
    ax2.set_title('Individual Method Scores (Normalized)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# --- CLI Entry Point ---
def main_cli():
    parser = argparse.ArgumentParser(description="Splash Detection Tools CLI")
    sub = parser.add_subparsers(dest='cmd')

    # Common zone configuration arguments for all commands
    def add_zone_args(p):
        p.add_argument('--zone', nargs=2, type=float, metavar=('TOP', 'BOTTOM'),
                       default=[0.7, 0.95], help='Splash zone vertical coordinates (normalized 0.0-1.0). Disables interactive selection.')
        p.add_argument('--bbox', nargs=4, type=float, metavar=('TOP', 'BOTTOM', 'LEFT', 'RIGHT'),
                       help='Splash zone bounding box (normalized coordinates: top bottom left right). Disables interactive selection.')
        p.add_argument('--interactive-zone', action='store_true',
                       help='Force interactive zone selection (this is the default behavior)')
        p.add_argument('--no-interactive-zone', action='store_true',
                       help='Disable interactive zone selection, use --zone coordinates instead')

    # Signals export mode
    p_export = sub.add_parser('export', help='Export detection signals to CSV')
    p_export.add_argument('video', help='Video file path')
    p_export.add_argument('output_csv', help='Output CSV path')
    p_export.add_argument('--weighted-fusion', action='store_true',
                         help='Use weighted fusion based on F1-scores instead of equal weighting')
    p_export.add_argument('--min-requirements', action='store_true',
                         help='Apply per-method minimum threshold requirements')
    p_export.add_argument('--temporal-consistency', action='store_true',
                         help='Apply temporal consistency (minimum splash duration)')
    p_export.add_argument('--min-duration', type=int, default=3,
                         help='Minimum splash duration in frames for temporal consistency (default: 3)')
    p_export.add_argument('--min-detectors', type=int, default=1,
                         help='Minimum number of detectors required to pass gate (default: 1)')
    p_export.add_argument('--temporal-bonus', type=float, default=0.8,
                         help='Bonus factor for frames outside temporal peaks (default: 0.8, use 1.0 for no penalty)')
    add_zone_args(p_export)

    # Tagging mode
    p_tag = sub.add_parser('tag', help='Tag frames for splash via GUI')
    p_tag.add_argument('video', help='Video file path')
    p_tag.add_argument('labels_out', help='Output CSV for labels')
    add_zone_args(p_tag)

    # Evaluation mode
    p_eval = sub.add_parser('evaluate', help='Evaluate signals against labels')
    p_eval.add_argument('signals_csv', help='CSV of signals')
    p_eval.add_argument('labels_csv', help='CSV of labels')
    p_eval.add_argument('--quantile', type=float, default=0.90,
                        help='Quantile for combined_score threshold (default: 0.90)')
    p_eval.add_argument('--sweep', action='store_true',
                        help='Run quantile sweep to test multiple thresholds')
    p_eval.add_argument('--auto-threshold', action='store_true',
                        help='Automatically select best threshold based on F1-score')
    p_eval.add_argument('--analyze-errors', action='store_true',
                        help='Analyze false positive and false negative cases')
    p_eval.add_argument('--per-method', action='store_true',
                        help='Analyze performance of individual detection methods')
    p_eval.add_argument('--plot', action='store_true',
                        help='Plot detection scores and thresholds over time')
    p_eval.add_argument('--all-analysis', action='store_true',
                        help='Run all analysis types (equivalent to --sweep --auto-threshold --analyze-errors --per-method --plot)')

    args = parser.parse_args()

    # Handle zone configuration for commands that need it
    def configure_zone(args):
        # Import interactive zone function if needed
        from splash_only_detector import get_splash_zone_interactive

        if hasattr(args, 'bbox') and args.bbox:
            # Use provided bounding box coordinates (takes precedence)
            zone_top, zone_bottom, zone_left, zone_right = args.bbox[0], args.bbox[1], args.bbox[2], args.bbox[3]
            print(f"📦 Using provided bounding box: top={zone_top:.3f}, bottom={zone_bottom:.3f}, left={zone_left:.3f}, right={zone_right:.3f}")
        elif hasattr(args, 'no_interactive_zone') and args.no_interactive_zone:
            # Use vertical zone with full width (interactive disabled)
            zone_top, zone_bottom = args.zone[0], args.zone[1]
            zone_left, zone_right = 0.0, 1.0  # Default to full width
            print(f"📊 Using vertical zone with full width: {zone_top:.3f} - {zone_bottom:.3f}")
        else:
            # Default behavior: Use interactive zone selection unless explicitly disabled
            # Check if --zone was explicitly provided vs default values
            zone_explicitly_provided = hasattr(args, 'zone') and args.zone != [0.7, 0.95]

            if zone_explicitly_provided:
                # User provided specific --zone coordinates, use them
                zone_top, zone_bottom = args.zone[0], args.zone[1]
                zone_left, zone_right = 0.0, 1.0  # Default to full width
                print(f"📊 Using specified zone: {zone_top:.3f} - {zone_bottom:.3f} (full width)")
            else:
                # Default to interactive zone selection
                try:
                    print(f"🎯 Interactive zone selection (default behavior)")
                    zone_top, zone_bottom, zone_left, zone_right = get_splash_zone_interactive(args.video)
                except Exception as e:
                    print(f"❌ Error during interactive zone selection: {e}")
                    print("🔄 Falling back to default zone")
                    zone_top, zone_bottom, zone_left, zone_right = 0.7, 0.95, 0.0, 1.0

        return zone_top, zone_bottom, zone_left, zone_right

    # Instantiate your existing SplashOnlyDetector
    from splash_only_detector import SplashOnlyDetector, DetectionConfig

    # Create configuration based on command
    if args.cmd == 'export':
        zone_top, zone_bottom, zone_left, zone_right = configure_zone(args)
        config = DetectionConfig(
            splash_zone_top_norm=zone_top,
            splash_zone_bottom_norm=zone_bottom,
            splash_zone_left_norm=zone_left,
            splash_zone_right_norm=zone_right
        )
        detector = SplashOnlyDetector(config)

        # Configure aggregator with enhanced options
        fusion_weights = None
        min_thresholds = None

        if args.weighted_fusion:
            # Use F1-based weights from analysis
            weights = {
                'flow': 0.884,
                'contour': 0.878,
                'diff': 0.872,
                'motion': 0.837
            }
            total = sum(weights.values())
            fusion_weights = {k: v/total for k, v in weights.items()}

        if args.min_requirements:
            min_thresholds = {
                'flow': 0.05,      # Optical flow minimum
                'contour': 100.0,  # Contour area minimum (further relaxed)
                'diff': 150.0,     # Frame difference minimum (further relaxed)
                'motion': 3.0      # Motion intensity minimum
            }

        agg = SplashSignalAggregator(
            detector,
            fusion_weights=fusion_weights,
            min_thresholds=min_thresholds,
            temporal_consistency=args.temporal_consistency,
            min_splash_duration=args.min_duration,
            min_detectors_required=args.min_detectors,
            temporal_bonus_factor=args.temporal_bonus
        )
        df = agg.process_video(args.video, config)
        export_signals_to_csv(df, args.output_csv)
    elif args.cmd == 'tag':
        zone_top, zone_bottom, zone_left, zone_right = configure_zone(args)
        config = DetectionConfig(
            splash_zone_top_norm=zone_top,
            splash_zone_bottom_norm=zone_bottom,
            splash_zone_left_norm=zone_left,
            splash_zone_right_norm=zone_right
        )
        detector = SplashOnlyDetector(config)
        tag_frames_cli(args.video, args.labels_out, detector, config)
    elif args.cmd == 'evaluate':
        # Handle --all-analysis flag
        if args.all_analysis:
            args.sweep = True
            args.auto_threshold = True
            args.analyze_errors = True
            args.per_method = True
            args.plot = True

        evaluate_signals(
            args.signals_csv,
            args.labels_csv,
            args.quantile,
            sweep_quantiles=args.sweep,
            auto_threshold=args.auto_threshold,
            analyze_errors=args.analyze_errors,
            per_method=args.per_method,
            plot_scores=args.plot
        )
    else:
        parser.print_help()

if __name__ == '__main__':
    main_cli()
