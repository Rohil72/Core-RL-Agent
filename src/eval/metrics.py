import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
from typing import Dict, Any, List, Optional
import os

sns.set_style("whitegrid")

def compute_classification_metrics(
    y_true: np.ndarray, 
    y_pred_prob: np.ndarray, 
    threshold: float = 0.5
) -> Dict[str, float]:
    """Compute precision, recall, F1, AUC."""
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    try:
        auc = roc_auc_score(y_true, y_pred_prob)
    except ValueError:
        auc = 0.5
        
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "threshold": float(threshold)
    }

def compute_calibration_metrics(
    y_true: np.ndarray, 
    y_pred_prob: np.ndarray, 
    n_bins: int = 10
) -> Dict[str, Any]:
    """Compute Brier score and calibration curve."""
    brier = brier_score_loss(y_true, y_pred_prob)
    
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_pred_prob, n_bins=n_bins, strategy='uniform')
    except (ValueError, IndexError):
        prob_true, prob_pred = np.array([]), np.array([])
        
    return {
        "brier_score": float(brier),
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist()
    }

def compute_stability_metric(
    embeddings: np.ndarray, 
    ticker_labels: List[str], 
    window_size: int = 10
) -> float:
    """
    Compute rolling-window centroid drift for same ticker.
    Returns average L2 distance between consecutive window centroids.
    """
    df = pd.DataFrame(embeddings)
    df['ticker'] = ticker_labels
    
    all_drifts = []
    
    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker].drop(columns=['ticker']).values
        
        if len(ticker_data) < window_size * 2:
            continue
            
        centroids = []
        for i in range(len(ticker_data) - window_size + 1):
            window = ticker_data[i:i + window_size]
            centroid = np.mean(window, axis=0)
            centroids.append(centroid)
            
        if len(centroids) > 1:
            centroids_arr = np.array(centroids)
            dists = np.linalg.norm(centroids_arr[1:] - centroids_arr[:-1], axis=1)
            all_drifts.extend(dists)
            
    return float(np.mean(all_drifts)) if all_drifts else 0.0

def compute_coverage(y_pred_flags: np.ndarray) -> float:
    """Fraction of cycles flagged."""
    if len(y_pred_flags) == 0:
        return 0.0
    return float(np.mean(y_pred_flags > 0))

def plot_calibration(
    prob_true: List[float], 
    prob_pred: List[float], 
    save_path: str
):
    """Generate reliability diagram."""
    if not prob_true or not prob_pred:
        return
        
    plt.figure(figsize=(7, 7))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title('Calibration Curve (Reliability Diagram)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_metrics_summary(metrics: Dict[str, float], save_path: str):
    """Bar plot of key metrics."""
    keys = ['precision', 'recall', 'f1', 'auc', 'coverage']
    values = [metrics.get(k, 0.0) for k in keys]
    
    plt.figure(figsize=(8, 5))
    plt.bar(keys, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    plt.ylim(0, 1.0)
    plt.ylabel('Score', fontsize=12)
    plt.title('Evaluation Metrics Summary', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def generate_all_plots(
    calibration_data: Dict[str, Any],
    metrics: Dict[str, float],
    save_dir: str,
    run_id: str
):
    """Generate and save all plots."""
    os.makedirs(save_dir, exist_ok=True)
    
    plot_calibration(
        calibration_data.get('prob_true', []),
        calibration_data.get('prob_pred', []),
        os.path.join(save_dir, f"{run_id}_calibration.png")
    )
    
    plot_metrics_summary(
        metrics,
        os.path.join(save_dir, f"{run_id}_metrics.png")
    )
