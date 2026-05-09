import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('results/plots', exist_ok=True)

environments = ['Wildlife', 'Traffic', 'Home', 'Classroom']
map50       = [0.8275,    0.5709,    0.5051,  0.3775]
precision   = [0.8754,    0.7071,    0.6371,  0.5449]
recall      = [0.8386,    0.5980,    0.5534,  0.4100]
f1          = [0.8566,    0.6480,    0.5923,  0.4679]

colors = ['#2ecc71', '#3498db', '#e67e22', '#e74c3c']

# Plot 1 — mAP per environment
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(environments, map50, color=colors, edgecolor='black', linewidth=0.5)
ax.set_title('mAP@0.5 per Environment', fontsize=14, fontweight='bold')
ax.set_ylabel('mAP@0.5')
ax.set_ylim(0, 1.0)
for bar, val in zip(bars, map50):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')
ax.axhline(y=0.5235, color='gray', linestyle='--', label='Overall mAP: 0.5235')
ax.legend()
plt.tight_layout()
plt.savefig('results/plots/map_per_environment.png', dpi=150)
plt.close()
print("Saved: map_per_environment.png")

# Plot 2 — Grouped bar chart P/R/F1
x = range(len(environments))
width = 0.25
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar([i - width for i in x], precision, width, label='Precision', color='#3498db', edgecolor='black', linewidth=0.5)
ax.bar([i         for i in x], recall,    width, label='Recall',    color='#2ecc71', edgecolor='black', linewidth=0.5)
ax.bar([i + width for i in x], f1,        width, label='F1-score',  color='#e67e22', edgecolor='black', linewidth=0.5)
ax.set_title('Precision, Recall and F1-score per Environment', fontsize=14, fontweight='bold')
ax.set_ylabel('Score')
ax.set_ylim(0, 1.0)
ax.set_xticks(list(x))
ax.set_xticklabels(environments)
ax.legend()
plt.tight_layout()
plt.savefig('results/plots/precision_recall_f1.png', dpi=150)
plt.close()
print("Saved: precision_recall_f1.png")

# Plot 3 — Overall metrics summary
metrics_names  = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall']
metrics_values = [0.5235,    0.3969,          0.6533,      0.5696]
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(metrics_names, metrics_values, color=['#9b59b6','#8e44ad','#3498db','#2ecc71'],
              edgecolor='black', linewidth=0.5)
ax.set_title('Overall Model Performance', fontsize=14, fontweight='bold')
ax.set_ylabel('Score')
ax.set_ylim(0, 1.0)
for bar, val in zip(bars, metrics_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('results/plots/overall_metrics.png', dpi=150)
plt.close()
print("Saved: overall_metrics.png")

print("\nAll plots saved to results/plots/")