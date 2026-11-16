
import csv
import numpy as np

rows = list(csv.DictReader(open("features_vp10.csv")))
t = np.array([float(r["time_seconds"]) for r in rows])
feats = ["audio_rms_db", "voice_energy_db", "onset_strength", "motion", "cut_change"]
X = {f: np.array([float(r[f]) for r in rows]) for f in feats}

windows = {"approved 593-648": (593, 648), "approved 639-696": (639, 696)}
W = 55  # window length in 1-second bins

for f in feats:
    x = X[f]
    sw = np.array([x[i:i + W].max() for i in range(len(x) - W + 1)])
    print(f"\n{f}: median={np.median(x):.3f}  p95={np.percentile(x, 95):.3f}  p99={np.percentile(x, 99):.3f}")
    for name, (a, b) in windows.items():
        m = (t >= a) & (t < b)
        wm = x[m].max()
        print(f"  {name}: window max={wm:.3f} -> beats {(sw < wm).mean() * 100:.0f}% of all {W}s windows in the file")
