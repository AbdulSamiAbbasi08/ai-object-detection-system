import csv
import os

ENVIRONMENTS = {
    'Traffic':   ['car', 'motorcycle', 'bus', 'truck', 'person', 'traffic light', 'bicycle'],
    'Home':      ['tv', 'laptop', 'cell phone', 'chair', 'bed', 'bottle', 'cup'],
    'Wildlife':  ['bird', 'cat', 'dog', 'horse', 'elephant', 'bear', 'zebra', 'giraffe'],
    'Classroom': ['backpack', 'book', 'clock', 'scissors', 'keyboard', 'mouse', 'handbag', 'potted plant'],
}

per_class = {
    'person':        {'mAP50': 0.676, 'P': 0.796, 'R': 0.694},
    'bicycle':       {'mAP50': 0.456, 'P': 0.656, 'R': 0.461},
    'car':           {'mAP50': 0.512, 'P': 0.760, 'R': 0.529},
    'motorcycle':    {'mAP50': 0.688, 'P': 0.747, 'R': 0.731},
    'bus':           {'mAP50': 0.860, 'P': 0.785, 'R': 0.889},
    'truck':         {'mAP50': 0.271, 'P': 0.471, 'R': 0.326},
    'traffic light': {'mAP50': 0.533, 'P': 0.735, 'R': 0.556},
    'tv':            {'mAP50': 0.595, 'P': 0.691, 'R': 0.618},
    'laptop':        {'mAP50': 0.700, 'P': 0.841, 'R': 0.682},
    'cell phone':    {'mAP50': 0.392, 'P': 0.583, 'R': 0.458},
    'chair':         {'mAP50': 0.342, 'P': 0.586, 'R': 0.424},
    'bed':           {'mAP50': 0.369, 'P': 0.616, 'R': 0.371},
    'bottle':        {'mAP50': 0.543, 'P': 0.640, 'R': 0.628},
    'cup':           {'mAP50': 0.595, 'P': 0.503, 'R': 0.693},
    'bird':          {'mAP50': 0.615, 'P': 0.880, 'R': 0.619},
    'cat':           {'mAP50': 0.806, 'P': 0.833, 'R': 0.850},
    'dog':           {'mAP50': 0.816, 'P': 0.878, 'R': 0.826},
    'horse':         {'mAP50': 0.908, 'P': 0.917, 'R': 0.917},
    'elephant':      {'mAP50': 0.693, 'P': 0.890, 'R': 0.696},
    'bear':          {'mAP50': 0.982, 'P': 0.810, 'R': 1.000},
    'zebra':         {'mAP50': 0.855, 'P': 0.832, 'R': 0.857},
    'giraffe':       {'mAP50': 0.945, 'P': 0.963, 'R': 0.944},
    'backpack':      {'mAP50': 0.271, 'P': 0.468, 'R': 0.281},
    'book':          {'mAP50': 0.102, 'P': 0.595, 'R': 0.094},
    'clock':         {'mAP50': 0.467, 'P': 0.730, 'R': 0.516},
    'scissors':      {'mAP50': 0.245, 'P': 0.500, 'R': 0.250},
    'keyboard':      {'mAP50': 0.629, 'P': 0.672, 'R': 0.646},
    'mouse':         {'mAP50': 0.665, 'P': 0.349, 'R': 0.667},
    'handbag':       {'mAP50': 0.238, 'P': 0.500, 'R': 0.312},
    'potted plant':  {'mAP50': 0.403, 'P': 0.545, 'R': 0.514},
}

print(f"\n{'='*60}")
print(f"PER-ENVIRONMENT METRICS")
print(f"{'='*60}")

env_results = {}
for env_name, env_classes in ENVIRONMENTS.items():
    maps, ps, rs = [], [], []
    for cls in env_classes:
        if cls in per_class:
            maps.append(per_class[cls]['mAP50'])
            ps.append(per_class[cls]['P'])
            rs.append(per_class[cls]['R'])

    avg_map = sum(maps) / len(maps)
    avg_p   = sum(ps)   / len(ps)
    avg_r   = sum(rs)   / len(rs)
    avg_f1  = (2 * avg_p * avg_r) / (avg_p + avg_r + 1e-6)
    env_results[env_name] = {'mAP50': avg_map, 'Precision': avg_p, 'Recall': avg_r, 'F1': avg_f1}

    print(f"\n{env_name}:")
    print(f"  mAP@0.5:   {avg_map:.4f}")
    print(f"  Precision: {avg_p:.4f}")
    print(f"  Recall:    {avg_r:.4f}")
    print(f"  F1-score:  {avg_f1:.4f}")

best_env  = max(env_results, key=lambda x: env_results[x]['mAP50'])
worst_env = min(env_results, key=lambda x: env_results[x]['mAP50'])
print(f"\n{'='*60}")
print(f"Best environment:  {best_env}  ({env_results[best_env]['mAP50']:.4f})")
print(f"Worst environment: {worst_env} ({env_results[worst_env]['mAP50']:.4f})")
print(f"Overall mAP@0.5:   0.5235")
print(f"Overall Precision: 0.6533")
print(f"Overall Recall:    0.5696")

os.makedirs('results', exist_ok=True)
with open('results/metrics.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Environment', 'mAP@0.5', 'Precision', 'Recall', 'F1-score'])
    for env, m in env_results.items():
        writer.writerow([env, f"{m['mAP50']:.4f}", f"{m['Precision']:.4f}",
                         f"{m['Recall']:.4f}", f"{m['F1']:.4f}"])
    writer.writerow(['Overall', '0.5235', '0.6533', '0.5696', 'N/A'])

print(f"\nMetrics saved to results/metrics.csv")