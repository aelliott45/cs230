import numpy as np
import matplotlib.pyplot as plt
import os

# --- User config ---
ann_file = './files/test_N14.csv'  # Update if needed
gt_dir = '/home/ec2-user/CircuitNet/IR_drop_training_set/IR_drop'
ground_threshold = 0.98           # regression threshold
pred_threshold = 0.98             # regression threshold
cls_threshold = 0.5         # classification threshold

# --- Pick a sample from the test CSV ---
example_num = 971
with open(ann_file, 'r') as f:
    for i in range(example_num):
        line = f.readline().strip()
    if ',' in line:
        feature, label = line.split(',')
    else:
        label = line

label_file = os.path.basename(label)

# --- List of (name, path) tuples ---
# plot_sources = [
#     ("Ground Truth", os.path.join(gt_dir, "label", label_file)),
#     ("Pretrained MAVI Prediction", 'work_dir/irdrop_mavi/test_result/' + label_file),
#     ("MAVI Prediction", 'work_dir/irdrop_mavi_trained2/test_result/' + label_file),
#     ("MAVI_TA_StaticFiLM_TemporalOnly Prediction", 'work_dir/irdrop_mavi_TA_StaticFiLM_TemporalOnly/test_result/' + label_file),
#     ("MAVI_TA_StaticFiLM_TemporalOnly_MultiTask Prediction", 'work_dir/MAVI_TA_StaticFiLM_TemporalOnly_MultiTask/test_result/' + label_file),
#     ("MAVI_TA_StaticFiLM_TemporalOnly_MultiTask_t98_clsw15 Prediction", 'work_dir/MAVI_TA_StaticFiLM_TemporalOnly_MultiTask_t98_clsw15/test_result/' + label_file),
#     ("MAVI_TA_StaticFiLM_TemporalOnly_MultiTask_t98_clsw40 Prediction", 'work_dir/MAVI_TA_StaticFiLM_TemporalOnly_MultiTask_t98_clsw40/test_result/' + label_file)
# ]
plot_sources = [
    # Ground Truth
    ("Ground Truth", os.path.join(gt_dir, "label", label_file)),

    # # Baselines
    # ("Pretrained MAVI Prediction",
    #  f"work_dir/irdrop_mavi/test_result/{label_file}"),

    ("Baseline MAVI Prediction",
     f"work_dir/irdrop_mavi_trained2/test_result/{label_file}"),

    ("MAVI + Laplacian Loss (0.5)",
     f"work_dir/irdrop_mavi_l1laploss_0.5_dec4/test_result/{label_file}"),

    # ("MAVI + Laplacian Loss (5.0)",
    #  f"work_dir/irdrop_mavi_l1laploss_5_dec4/test_result/{label_file}"),

    ("MAVI + SE + Laplacian Loss (0.5)",
     f"work_dir/irdrop_maviSE_l1laploss_0.5/test_result/{label_file}"),

    # ("MAVI + Laplacian Loss (prediction-only smoothing, 0.5)",
    #  f"work_dir/irdrop_maviSE_l1laploss_1_pred/test_result/{label_file}"),

    # ("MAVI + Laplacian Loss (1)",

    #  f"work_dir/irdrop_maviSE_l1laploss_1_match/test_result/{label_file}"),

    # Temporal Attention + Static FiLM variants
    ("MAVI_TA_StaticFiLM_TemporalOnly Prediction",
     f"work_dir/irdrop_mavi_TA_StaticFiLM_TemporalOnly/test_result/{label_file}"),

    # # Multi-Task variants
    # ("MultiTask TA+StaticFiLM Prediction",
    #  f"work_dir/MAVI_TA_StaticFiLM_TemporalOnly_MultiTask/test_result/{label_file}"),

    # ("MultiTask TA+StaticFiLM (t=0.98, cls weight=0.15)",
    #  f"work_dir/MAVI_TA_StaticFiLM_TemporalOnly_MultiTask_t98_clsw15/test_result/{label_file}"),

    ("MultiTask TA+StaticFiLM (t=0.98, cls weight=0.40)",
     f"work_dir/MAVI_TA_StaticFiLM_TemporalOnly_MultiTask_t98_clsw40/test_result/{label_file}")
]


# --- Load regression arrays + try classification arrays ---
arrays = []
cls_arrays = []
names = [name for name, _ in plot_sources]

for name, path in plot_sources:
    # regression
    arr = np.load(path).squeeze()
    arrays.append(arr)

    # try classification file
    cls_path = path.replace('.npy', '_cls.npy')
    if os.path.exists(cls_path):
        cls_arr = np.load(cls_path).squeeze()
        cls_arrays.append(cls_arr)
    else:
        cls_arrays.append(None)

# optional highlight transform
arrays = [np.power(arr, 4) for arr in arrays]

vmin, vmax = arrays[0].min(), arrays[0].max()

# regression masks
reg_masks = [(arr > pred_threshold).astype(np.uint8) for arr in arrays]
reg_masks[0] = (arrays[0] > ground_threshold).astype(np.uint8)  

# classifier masks
cls_masks = [
    (cls_arr > cls_threshold).astype(np.uint8) if cls_arr is not None else None
    for cls_arr in cls_arrays
]

# --- Make 4-row plot ---
fig, axs = plt.subplots(4, len(arrays), figsize=(5 * len(arrays), 28))

# Row 1: Regression heatmaps
# for ax, name, arr in zip(axs[0], names, arrays):
#     ax.imshow(arr, cmap='viridis', vmin=vmin, vmax=vmax)
#     ax.set_title(f"{name}\nreg map\n(min={arr.min():.4f}, max={arr.max():.4f})")
#     ax.axis('off')

# Row 2: Regression masks
for ax, name, mask in zip(axs[1], names, reg_masks):
    if name == "MultiTask TA+StaticFiLM (t=0.98, cls weight=0.40)":
        if cls_arr is not None:
            ax.imshow(cls_arr, cmap='hot', vmin=0, vmax=1)
        else:
            ax.text(0.5, 0.5, "No cls file", ha='center', va='center')
        continue
    ax.set_title(f"{name}\nreg > {pred_threshold}")
    ax.axis('off')
    ys, xs = np.where(mask)
    ax.scatter(xs, ys, s=1, c='red')
    ax.set_xlim([0, mask.shape[1]])
    ax.set_ylim([mask.shape[0], 0])

# Row 3: Classifier heatmaps
for ax, name, cls_arr in zip(axs[2], names, cls_arrays):
    ax.axis('off')
    if cls_arr is None:
        ax.set_title(f"{name}\n(no cls file)")
        continue
    ax.imshow(cls_arr, cmap='hot', vmin=0, vmax=1)
    # ax.set_title(f"{name}\ncls prob\n(min={cls_arr.min():.4f}, max={cls_arr.max():.4f})")
    ax.set_title(f"{name}\ncls prob")
# Row 4: Classifier masks
for ax, name, cls_mask in zip(axs[3], names, cls_masks):
    ax.axis('off')
    if cls_mask is None:
        ax.set_title(f"{name}\n(no cls file)")
        continue
    ax.set_title(f"{name}\ncls > {cls_threshold}")
    ys, xs = np.where(cls_mask)
    ax.scatter(xs, ys, s=1, c='red')
    ax.set_xlim([0, cls_mask.shape[1]])
    ax.set_ylim([cls_mask.shape[0], 0])

plt.tight_layout()
outfile = 'comparison_heatmap_regcls.png'
plt.savefig(outfile, dpi=200)
plt.show()

print("Saved:", outfile)
