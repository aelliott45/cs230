plot_sources = [
    ("Ground Truth", os.path.join(gt_dir, "label", label_file)),
    ("Pretrained MAVI Prediction", 'work_dir/irdrop_mavi/test_result/' + label_file),
    ("MAVI Prediction", 'work_dir/irdrop_mavi_trained2/test_result/' + label_file),
    ("MAVI_LAP 0.5 match Prediction", 'work_dir/irdrop_mavi_l1laploss_0.5_dec4/test_result/' + label_file),
    ("MAVI_SE Prediction", 'work_dir/irdrop_maviSE_l1laploss_0.5/test_result/' + label_file),
    ("MAVI_LAP 5 match Prediction", 'work_dir/irdrop_mavi_l1laploss_5_dec4/test_result/' + label_file),
    ("MAVI_TA_StaticFiLM_TemporalOnly Prediction", 'work_dir/irdrop_mavi_TA_StaticFiLM_TemporalOnly/test_result/' + label_file),
    ("MAVI_LAP 0.5 pred Prediction", 'work_dir/irdrop_maviSE_l1laploss_1_pred/test_result/' + label_file)
    # Add more tuples here as needed
]


import numpy as np
import matplotlib.pyplot as plt
import os

# --- User config ---
ann_file = './files/test_N14.csv'  # Update if needed
gt_dir = '/home/ec2-user/CircuitNet/IR_drop_training_set/IR_drop'

# --- Pick a sample from the test CSV ---
example_num = 293
with open(ann_file, 'r') as f:
    for i in range(example_num):
        line = f.readline().strip()
    if ',' in line:
        feature, label = line.split(',')
    else:
        label = line

label_file = os.path.basename(label)

# --- List of (name, path) tuples ---
plot_sources = [
    ("Ground Truth", os.path.join(gt_dir, "label", label_file)),
    ("Pretrained MAVI Prediction", 'work_dir/irdrop_mavi/test_result/' + label_file),
    ("MAVI Prediction", 'work_dir/irdrop_mavi_trained2/test_result/' + label_file),
    ("MAVI_LAP 0.5 match Prediction", 'work_dir/irdrop_mavi_l1laploss_0.5_dec4/test_result/' + label_file),
    ("MAVI_SE Prediction", 'work_dir/irdrop_maviSE_l1laploss_0.5/test_result/' + label_file),
    ("MAVI_LAP 5 match Prediction", 'work_dir/irdrop_mavi_l1laploss_5_dec4/test_result/' + label_file),
    ("MAVI_TA_StaticFiLM_TemporalOnly Prediction", 'work_dir/irdrop_mavi_TA_StaticFiLM_TemporalOnly/test_result/' + label_file),
    ("MAVI_LAP 0.5 pred Prediction", 'work_dir/irdrop_maviSE_l1laploss_1_pred/test_result/' + label_file)
    # Add more tuples here as needed
]

# --- Load arrays ---
arrays = []
for name, path in plot_sources:
    arr = np.load(path)
    arr = np.squeeze(arr)
    
    arrays.append(arr)

arrays = [np.power(arr, 4) for arr in arrays]  
vmin, vmax = arrays[0].min(), arrays[0].max()

# --- Plot and save PNG ---
fig, axs = plt.subplots(1, len(arrays), figsize=(5 * len(arrays), 8))

if len(arrays) == 1:
    axs = [axs]  # Ensure axs is iterable

for ax, (name, arr) in zip(axs, zip([n for n, _ in plot_sources], arrays)):
    ax.imshow(arr, cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_title(name)
    ax.axis('off')

plt.tight_layout()
plt.savefig('comparison_heatmap.png', dpi=200)
plt.show()
print('Saved: comparison_heatmap.png')
