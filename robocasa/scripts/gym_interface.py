from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS
from robocasa.utils.env_utils import create_env, run_random_rollouts
import robocasa.utils.robomimic.robomimic_env_utils as EnvUtils
import numpy as np
from robosuite.utils.camera_utils import get_real_depth_map
import cv2
import os
from tqdm import tqdm

# choose random task
# env_name = np.random.choice(list(ALL_KITCHEN_ENVIRONMENTS))
env_name = "PnPCounterToSink"

env = EnvUtils.create_env(
    env_type=1,
    env_name=env_name,
    # render_onscreen=False,
    seed=None, # set seed=None to run unseeded
    camera_names=["robot0_birdview", "robot0_agentview_left"],
    camera_widths=256,
    camera_heights=256,
)

# reset the environment
env.reset()

# get task language
lang = env.get_ep_meta()["lang"]
print("Instruction:", lang)

# set cam
# cam_name = "robot0_birdview"
cam_name = "robot0_agentview_left"

# output video path
out_path = f"outputs/{cam_name}.mp4"
writer = None

for i in tqdm(range(50)):
    action = np.random.randn(*env.action_spec[0].shape) * 0.1
    obs, reward, done, info = env.step(action)  # take action in the environment

    image = obs[f"{cam_name}_image"]  # get birdview image
    depth = obs[f"{cam_name}_depth"]  # get birdview depth
    depth = get_real_depth_map(env.sim, depth)  # convert depth to real depth in meters

    # --- Normalize shapes: make image (H,W,3) and depth (H,W) ---
    img = image
    # if image is (H,W,3) already, keep it
    # convert float images to uint8 [0,255]
    if img.dtype != np.uint8:
        # assume float in [0,1] or arbitrary range; clip then scale
        img = np.asarray(img, dtype=np.float32)
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255).astype(np.uint8)

    d = depth
    d = d[..., 0]
    d = np.asarray(d, dtype=np.float32)

    # normalize depth per-frame to [0,255]
    if np.isfinite(d).any():
        dmin = float(np.nanmin(d))
        dmax = float(np.nanmax(d))
    else:
        dmin, dmax = 0.0, 1.0
    if dmax - dmin < 1e-6:
        depth_norm = np.zeros_like(d, dtype=np.uint8)
    else:
        depth_norm = ((d - dmin) / (dmax - dmin) * 255.0).astype(np.uint8)

    # apply colormap (returns BGR)
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

    # ensure same H,W for image and depth_color
    if depth_color.shape[:2] != img.shape[:2]:
        depth_color = cv2.resize(depth_color, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

    # convert img RGB->BGR if needed (cv2 uses BGR)
    # Heuristic: if first channel looks like R channel (mean differs), assume RGB
    if img.shape[2] == 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img

    # concatenate horizontally: [rgb | depth_heatmap]
    concat = np.concatenate([img_bgr, depth_color], axis=1)

    # initialize writer after knowing frame size
    if writer is None:
        h, w = concat.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 30
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    writer.write(concat)

    if done:
        break

if writer is not None:
    writer.release()
    print(f"Saved video to {os.path.abspath(out_path)}")
