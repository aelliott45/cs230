import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# --- Config ---
ann_file = './files/test_N14.csv'
gt_dir = '/home/ec2-user/CircuitNet/IR_drop_training_set/IR_drop'
ground_threshold = 0.98
cls_threshold = 0.5

# --- Pick the same sample you used ---
example_num = 971  # or whichever index you want

# --- Get the filename from ann_file ---
with open(ann_file, 'r') as f:
    for i in range(example_num):
        line = f.readline().strip()
    if ',' in line:
        feature, label = line.split(',')
    else:
        label = line

label_file = os.path.basename(label)

# --- Paths for your model ---
model_name = "MAVI_TA_StaticFiLM_TemporalOnly_MultiTask_t98_clsw40"
pred_dir = f"work_dir/{model_name}/test_result"

reg_path = os.path.join(pred_dir, label_file)              # regression
cls_path = reg_path.replace('.npy', '_cls.npy')            # classification

# --- Load arrays ---
gt_reg = np.load(os.path.join(gt_dir, "label", label_file)).squeeze()
reg_pred = np.load(reg_path).squeeze()
cls_pred = np.load(cls_path).squeeze()   # shape [H, W], prob

# --- Construct ground-truth hotspot mask ---
gt_cls = (gt_reg > ground_threshold).astype(int)

# --- Predicted hotspot mask (classification head) ---
pred_cls = (cls_pred >= cls_threshold).astype(int)

# --- Flatten for confusion matrix ---
y_true = gt_cls.flatten()
y_pred = pred_cls.flatten()

# --- Compute confusion matrix ---
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

# --- Plot (publication-ready) ---
plt.figure(figsize=(4.5, 4.5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Pred 0", "Pred 1"],
    yticklabels=["True 0", "True 1"],
    cbar=False
)
plt.xlabel("Prediction")
plt.ylabel("Ground Truth")
plt.title(f"Confusion Matrix — {model_name}")
plt.tight_layout()

# Save a paper-ready PNG or PDF
plt.savefig("confusion_matrix_multitask.png", dpi=300, bbox_inches="tight")
plt.savefig("confusion_matrix_multitask.pdf", bbox_inches="tight")

plt.show()

print("Confusion matrix:\n", cm)
