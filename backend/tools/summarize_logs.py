"""
Script to summarize prediction logs.
"""
import json
from pathlib import Path
from collections import Counter, defaultdict
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.config import PREDICTION_LOG_FILE


def summarize_logs():
    """Summarize prediction logs."""
    if not PREDICTION_LOG_FILE.exists():
        print(f"Log file not found: {PREDICTION_LOG_FILE}")
        return
    
    predictions = []
    with open(PREDICTION_LOG_FILE, 'r') as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
    
    if not predictions:
        print("No predictions found in log file.")
        return
    
    print("="*60)
    print("PREDICTION LOG SUMMARY")
    print("="*60)
    print(f"Total predictions: {len(predictions)}\n")
    
    # Count by dialect
    dialect_counts = Counter(p['predicted_dialect'] for p in predictions)
    print("Predictions by Dialect:")
    for dialect, count in dialect_counts.most_common():
        pct = (count / len(predictions)) * 100
        print(f"  {dialect:20s}: {count:4d} ({pct:5.2f}%)")
    
    print()
    
    # Count by model
    model_counts = Counter(p['model_name'] for p in predictions)
    print("Predictions by Model:")
    for model, count in model_counts.most_common():
        pct = (count / len(predictions)) * 100
        print(f"  {model:20s}: {count:4d} ({pct:5.2f}%)")
    
    print()
    
    # Count by window mode
    window_counts = Counter(p['window_mode'] for p in predictions)
    print("Predictions by Window Mode:")
    for mode, count in window_counts.most_common():
        pct = (count / len(predictions)) * 100
        print(f"  {mode:20s}: {count:4d} ({pct:5.2f}%)")
    
    print()
    
    # Average confidence
    avg_confidence = sum(p['confidence'] for p in predictions) / len(predictions)
    print(f"Average Confidence: {avg_confidence:.4f} ({avg_confidence*100:.2f}%)")
    
    # Average duration
    avg_duration = sum(p['duration_sec'] for p in predictions) / len(predictions)
    print(f"Average Audio Duration: {avg_duration:.2f} seconds")
    
    print("="*60)


if __name__ == "__main__":
    summarize_logs()

