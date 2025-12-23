import h5py
import json
import os
from PIL import Image
import numpy as np

f = h5py.File("datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToSink/mg/2024-05-04-22-14-06_and_2024-05-07-07-40-17/demo_im128_ep10_cam2+1.hdf5", "r")
demo = f["data"]["demo_0"]                        # access demo 5
obs = demo["obs"]                                 # obervations across all timesteps

# 定义要提取的视图及其名称
cam_candidates = [
    "robot0_agentview_left_image",
    "robot0_agentview_right_image",
    "robot0_eye_in_hand_image",
    "robot0_birdview_image",
]

# 创建输出目录
out_dir = "./outputs"

# 提取并保存第一帧
for key in cam_candidates:
    arr = obs[key][0]  # 第一帧
    # 归一化到 0-255 并转为 uint8（若数据不是 uint8）
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    save_path = os.path.join(out_dir, f"{key}_frame0.png")
    img.save(save_path)
    print(f"Saved: {save_path}")

ep_meta = json.loads(demo.attrs["ep_meta"])       # get meta data for episode
lang = ep_meta["lang"]                            # get language instruction for episode
print("Instruction:", lang)
f.close()