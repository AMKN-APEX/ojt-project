import os
import numpy as np
import h5py
import torch

TARGET_NAMES = ["m_train", "m_val", "m_test", "kappa_train", "kappa_val", "kappa_test"]


# --- targetのmat2pt ---
for TARGET_NAME in TARGET_NAMES:
    mat_target_path = f"/mnt/c/Users/onion/Documents/data/{TARGET_NAME}.mat" # MATLABファイルのパス
    pt_target_path = f"/mnt/c/Users/onion/Documents/processed_data/target/{TARGET_NAME}.pt" # 保存先の.ptファイルのパス

    # pathのディレクトリ部分が存在しない場合は作成、存在してもエラーにしない
    os.makedirs(os.path.dirname(pt_target_path), exist_ok=True)

    # MATLABファイルを読み込み、指定された変数をPyTorchのテンソルに変換して保存
    with h5py.File(mat_target_path, "r") as f:
        keys = list(f.keys())
        data = f[keys[0]]
        data = np.array(data).T
        data = torch.tensor(data)
        torch.save(data, pt_target_path)

# --- structureのmat2pt ---
mat_path = f"/mnt/c/Users/onion/Documents/data/structures_train/structure_1.mat" # MATLABファイルのパス
pt_path = f"/mnt/c/Users/onion/Documents/processed_data/train/structure_1.pt" # 保存先の.ptファイルのパス

os.makedirs(os.path.dirname(pt_path), exist_ok=True)

with h5py.File(mat_path, "r") as f:
    keys = list(f.keys())
    data = f[keys[0]]
    data = np.array(data).T
    data = torch.tensor(data)
    torch.save(data, pt_path)