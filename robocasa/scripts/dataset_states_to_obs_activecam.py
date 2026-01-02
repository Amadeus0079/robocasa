"""
Script to extract observations from low-dimensional simulation states in a robocasa dataset.
Adapted from robomimic's dataset_states_to_obs.py script.
"""
import os
import json
import h5py
import argparse
import numpy as np
from copy import deepcopy
import multiprocessing
import queue
import time
import traceback
import torch

import robocasa.utils.robomimic.robomimic_tensor_utils as TensorUtils
import robocasa.utils.robomimic.robomimic_env_utils as EnvUtils
import robocasa.utils.robomimic.robomimic_dataset_utils as DatasetUtils

# from robomimic.utils.log_utils import log_warning
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_camera_extrinsic_matrix, get_camera_transform_matrix, get_real_depth_map

from collections import OrderedDict
from typing import Tuple, Union, Optional
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
import random

D_RANGE = (0.5, 1.0)
A_RANGE = (90.0, 270.0)
E_RANGE = (-90.0, 0.0)

def _nearest_rotation(R: np.ndarray) -> np.ndarray:
    """
    Project a 3x3 matrix to the nearest proper rotation (SO(3)) using SVD.
    Ensures det=+1.
    """
    U, S, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:
        # Fix reflection by flipping the last column of U
        U[:, -1] *= -1
        R_ortho = U @ Vt
    return R_ortho

def sample_sub_range(
    parent_range: Tuple[float, float],
    window_size: float,
    random_seed: int = None
) -> Tuple[float, float]:
    """
    在一个给定的父范围 [min_parent, max_parent] 内，随机采样一个固定大小的子范围。

    Args:
        parent_range (Tuple[float, float]): 父范围的 (min_parent, max_parent) 元组。
        window_size (float): 要采样的子范围的固定长度。
        random_seed (int, optional): 随机数种子，用于重现性。默认为 None。

    Returns:
        Tuple[float, float]: 采样的子范围 (min_child, max_child)。

    Raises:
        ValueError: 如果 window_size 大于父范围的总长度。
    """
    if random_seed is not None:
        random.seed(random_seed)

    min_parent, max_parent = parent_range
    
    if window_size <= 0:
        raise ValueError("window_size 必须为正数。")

    if window_size > (max_parent - min_parent):
        raise ValueError(
            f"window_size ({window_size}) 不能大于父范围的总长度 "
            f"({max_parent - min_parent})。请减小 window_size。"
        )
    
    # 子范围的起始点可以在 [min_parent, max_parent - window_size] 之间随机选择
    # 这个范围确保了子范围的结束点不会超出 max_parent
    min_child_possible_start = min_parent
    max_child_possible_start = max_parent - window_size

    # 在这个允许的范围内随机采样 min_child
    min_child = random.uniform(min_child_possible_start, max_child_possible_start)
    
    max_child = min_child + window_size
    
    return min_child, max_child

def invert_camera_extrinsic_matrix(R_extrinsic: np.ndarray):
    """
    Invert the transformation performed by:

        pose = T.make_pose(camera_pos, camera_rot)  # 4x4, world_T_camera(mujoco)
        C = diag([1, -1, -1, 1])
        R_extrinsic = pose @ C

    Given R_extrinsic (4x4), recover:
      - camera_pos (3,)
      - camera_rot (3,3)  in MuJoCo's native convention (sim.data.cam_xpos / cam_xmat).

    Parameters
    ----------
    R_extrinsic : np.ndarray
        4x4 homogeneous matrix returned by get_camera_extrinsic_matrix.

    Returns
    -------
    camera_pos : np.ndarray
        Shape (3,), MuJoCo camera position in world frame (cam_xpos).
    camera_rot : np.ndarray
        Shape (3,3), MuJoCo camera rotation matrix in world frame (cam_xmat).

    Notes
    -----
    - The axis correction matrix C is its own inverse (C == C^{-1}), so pose = R_extrinsic @ C.
    - Translation is unaffected by right-multiplying C, so cam position is simply pose[:3, 3].
    - A small numerical orthonormalization is applied to the recovered rotation.
    """
    R_extrinsic = np.asarray(R_extrinsic, dtype=float)
    if R_extrinsic.shape != (4, 4):
        raise ValueError(f"R_extrinsic must be 4x4, got shape {R_extrinsic.shape}")

    # Optional sanity check for homogeneous transform bottom row
    bottom_expected = np.array([0.0, 0.0, 0.0, 1.0])
    if not np.allclose(R_extrinsic[3, :], bottom_expected, atol=1e-7):
        raise ValueError(
            f"R_extrinsic bottom row should be [0,0,0,1], got {R_extrinsic[3, :]}"
        )

    # The same axis correction used in the forward path
    C = np.diag([1.0, -1.0, -1.0, 1.0])

    # Undo the right-multiplication by C (C is its own inverse)
    pose = R_extrinsic @ C  # pose = world_T_camera(mujoco)

    camera_rot_raw = pose[:3, :3]
    camera_pos = pose[:3, 3].copy()

    # Numerical safeguard: project rotation to SO(3)
    camera_rot = _nearest_rotation(camera_rot_raw)

    return camera_pos, camera_rot

def compose_camera_extrinsic(lookat: np.ndarray, distance: float, azimuth: float, elevation: float) -> np.ndarray:
    '''
    Calculates the camera extrinsic matrix (T_world_camera, camera pose in world coordinates)
    based on lookat point, distance, azimuth, and elevation, respecting specific coordinate systems.

    World coordinate system:
        Z-axis: Vertical (up)
        Y-axis: Horizontal left
        X-axis: Horizontal forward

    Camera coordinate system:
        Z-axis: Forward
        Y-axis: Downward
        X-axis: Rightward

    Input Parameters:
        azimuth (float): Rotation around world's vertical Z-axis, in degrees.
                         0 deg looks +X (forward), 90 deg looks +Y (left) (counter-clockwise from +X).
        elevation (float): Rotation around camera's local X-axis, in degrees.
                           Negative values move camera up, positive down (e.g., -45 deg means camera looks up 45 deg).

    Args:
        lookat (np.ndarray): (x, y, z) - Point camera is looking at in world coordinates.
        distance (float): Distance from camera to the lookat point.
        azimuth (float): Azimuth angle in degrees.
        elevation (float): Elevation angle in degrees.

    Returns:
        np.ndarray: 4x4 camera pose (T_world_camera) which transforms points from camera frame to world frame.
    '''
    lookat = np.asarray(lookat, dtype=np.float32)

    # World coordinate system up vector (Z-axis is vertical)
    world_up_vector = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # Convert angles to radians
    azimuth_rad = np.deg2rad(azimuth)
    elevation_rad = np.deg2rad(elevation)

    # Adjust elevation for "negative value moves camera up" convention
    # If elevation is -45 (up 45), effective_elevation_rad becomes +45 deg (pi/4)
    effective_elevation_rad = -elevation_rad

    # 1. Calculate camera position in world coordinates (T_world_camera's translation)
    # Based on spherical coordinates relative to 'lookat' point
    # azimuth=0 -> +X, elevation=0 -> horizontal
    cam_pos_x = distance * np.cos(effective_elevation_rad) * np.cos(azimuth_rad)
    cam_pos_y = distance * np.cos(effective_elevation_rad) * np.sin(azimuth_rad)
    cam_pos_z = distance * np.sin(effective_elevation_rad)
    camera_pos_world = lookat + np.array([cam_pos_x, cam_pos_y, cam_pos_z], dtype=np.float32)

    # 2. Build T_world_camera (camera pose in world)
    # Calculate the camera's axes in world coordinates
    # Camera Z-axis points from camera_pos_world towards lookat (Forward)
    camera_z_world = (lookat - camera_pos_world)
    camera_z_world /= np.linalg.norm(camera_z_world)

    # Camera X-axis (Rightward) in world coordinates
    # For robust cross product, especially when camera_z_world is aligned with world_up_vector
    # camera_x_world = np.cross(camera_z_world, world_up_vector) would give a vector to the *left*
    # if camera_z_world is in +X and world_up_vector is +Z (cross(X,Z) = -Y, which is right in new coord)
    # Let's test with new definition: cross([1,0,0], [0,0,1]) = [0, -1, 0] (World -Y, which is Right)
    camera_x_world = np.cross(camera_z_world, world_up_vector)
    # Handle gimbal lock case: camera_z_world is aligned with world_up_vector
    if np.linalg.norm(camera_x_world) < 1e-6:
        # Camera is looking straight up or straight down.
        # Arbitrarily pick world Y-axis as camera X-axis
        if camera_z_world[2] > 0: # Looking up (+Z)
             camera_x_world = np.array([0.0, 1.0, 0.0], dtype=np.float32) # World +Y (Left)
        else: # Looking down (-Z)
             camera_x_world = np.array([0.0, -1.0, 0.0], dtype=np.float32) # World -Y (Right)
    camera_x_world /= np.linalg.norm(camera_x_world)

    # Camera Y-axis (Downward) in world coordinates
    # In a right-handed system, cross(X, Z) = -Y
    # Here, we want Camera Y-axis to be Downward, which is opposite of the Up vector derived from cross(X,Z)
    camera_y_world = np.cross(camera_x_world, camera_z_world)
    # This cross product gives a vector that is UPWARD relative to the camera's XZ plane.
    # Since camera Y-axis is defined as DOWNWARD, we need to negate this vector.
    camera_y_world = -camera_y_world
    camera_y_world /= np.linalg.norm(camera_y_world)

    # Construct rotation matrix for T_world_camera
    # Columns are Camera X, Y, Z axes in World coordinates
    rot_matrix_world_camera = np.eye(3, dtype=np.float32)
    rot_matrix_world_camera[:, 0] = camera_x_world
    rot_matrix_world_camera[:, 1] = camera_y_world
    rot_matrix_world_camera[:, 2] = camera_z_world

    # Build T_world_camera homogeneous matrix
    t_world_camera = np.eye(4, dtype=np.float32)
    t_world_camera[:3, :3] = rot_matrix_world_camera
    t_world_camera[:3, 3] = camera_pos_world

    return t_world_camera


def create_camera_intrinsic(h: int, w: int, fovy: float) -> np.ndarray:
    '''
    Generates a 3x3 camera intrinsic matrix based on image dimensions and vertical field of view.

    Assumptions:
    - Principal point (cx, cy) is at the center of the image (w/2, h/2).
    - Pixels are square, meaning fx = fy (or fx = fy * aspect_ratio to maintain physical focal length).
      Here, we will calculate fy directly from fovy and h, then set fx = fy.

    Args:
        h (int): Image height in pixels.
        w (int): Image width in pixels.
        fovy (float): Vertical Field of View in degrees.

    Returns:
        np.ndarray: A 3x3 camera intrinsic matrix.
    '''
    if not isinstance(h, int) or h <= 0:
        raise ValueError("Image height (h) must be a positive integer.")
    if not isinstance(w, int) or w <= 0:
        raise ValueError("Image width (w) must be a positive integer.")
    if not isinstance(fovy, (float, int)) or fovy <= 0 or fovy >= 180:
        raise ValueError("Vertical FOV (fovy) must be a positive float/int less than 180 degrees.")

    # Convert fovy from degrees to radians
    fovy_rad = np.deg2rad(fovy)

    # Calculate focal length fy
    # tan(fovy/2) = (h/2) / fy
    # fy = (h / 2) / tan(fovy / 2)
    # Ensure fovy_rad / 2 is not too close to 0 or pi for tan
    if np.abs(np.tan(fovy_rad / 2)) < 1e-6:
        raise ValueError("fovy is too small, leading to division by zero or very large focal length.")
    
    fy = (h / 2.0) / np.tan(fovy_rad / 2.0)

    # Assuming square pixels, so fx = fy
    fx = fy

    # Principal point (center of the image)
    cx = w / 2.0
    cy = h / 2.0

    # Construct the intrinsic matrix
    intrinsic_matrix = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    return intrinsic_matrix


def sample_cameras(
    lookat: np.ndarray,
    d_range: Tuple[float, float] = (0.5, 1.5),
    a_range: Tuple[float, float] = (-180.0, 180.0),
    e_range: Tuple[float, float] = (-90.0, 90.0),
    num_cameras: int = 1,
    random_seed: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Randomly samples camera poses (T_world_camera) within specified ranges using compose_camera_extrinsic.

    Args:
        lookat (np.ndarray): (x, y, z) - Point camera is looking at in world coordinates.
        d_range (Tuple[float, float]): (min_distance, max_distance) for camera distance.
        a_range (Tuple[float, float]): (min_azimuth, max_azimuth) for camera azimuth in degrees.
        e_range (Tuple[float, float]): (min_elevation, max_elevation) for camera elevation in degrees.
        num_cameras (int): The number of camera poses to sample.
        random_seed (int, optional): Seed for the random number generator for reproducibility. Defaults to None.

    Returns:
        np.ndarray: (num_cameras, 3) array of camera states (distance, azimuth, elevation). (range from -1 to 1)
        np.ndarray: (num_cameras, 4, 4) array of camera poses (T_world_camera).
    '''
    if random_seed is not None:
        np.random.seed(random_seed)

    sampled_states = []
    sampled_poses = []
    for _ in range(num_cameras):
        # 随机采样距离
        norm_distance = np.random.random()
        distance = (d_range[1] - d_range[0]) * norm_distance + d_range[0]
        # 随机采样方位角
        norm_azimuth = np.random.random()
        azimuth = (a_range[1] - a_range[0]) * norm_azimuth + a_range[0]
        # 随机采样俯仰角
        norm_elevation = np.random.random()
        elevation = (e_range[1] - e_range[0]) * norm_elevation + e_range[0]

        # 调用您提供的函数来计算相机位姿
        camera_state = np.array((norm_distance, norm_azimuth, norm_elevation))
        camera_pose = compose_camera_extrinsic(lookat, distance, azimuth, elevation)
        sampled_states.append(camera_state)
        sampled_poses.append(camera_pose)

    return np.array(sampled_states, dtype=np.float32), np.array(sampled_poses, dtype=np.float32)

def set_camera_pose_world(sim, cam_id, pos_world, rot_world_3x3):
    # 写回 MuJoCo 相机位置与四元数（w, x, y, z）
    quat_wxyz = R.from_matrix(rot_world_3x3).as_quat()  # returns (x, y, z, w)
    # 转为 (w, x, y, z)
    quat_wxyz = np.roll(quat_wxyz, 1)
    sim.model.cam_pos[cam_id] = np.asarray(pos_world, dtype=np.float64)
    sim.model.cam_quat[cam_id] = np.asarray(quat_wxyz, dtype=np.float64)
    sim.forward()

def extract_trajectory(
    env,
    initial_state,
    states,
    actions,
    done_mode,
    add_datagen_info=False,
    interval=1,
):
    """
    Helper function to extract observations, rewards, and dones along a trajectory using
    the simulator environment.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load to extract information
        actions (np.array): array of actions
        done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a
            success state. If 1, done is 1 at the end of each trajectory.
            If 2, do both.
    """
    assert states.shape[0] == actions.shape[0]

    # load the initial state
    env.reset()
    obs = env.reset_to(initial_state)

    # get updated ep meta in case it's been modified
    ep_meta = env.env.get_ep_meta()
    initial_state["ep_meta"] = json.dumps(ep_meta, indent=4)

    traj = dict(
        obs=[],
        next_obs=[],
        rewards=[],
        dones=[],
        actions=[],
        actions_abs=[],
        states=[],
        initial_state_dict=initial_state,
        datagen_info=[],
        cam_info=[],
    )
    traj_len = states.shape[0]

    index = list(range(0, traj_len, interval))

    # Limit the active view range
    distance = np.random.uniform(D_RANGE[0], D_RANGE[1])
    azimuth = np.random.uniform(A_RANGE[0], A_RANGE[1])
    elevation = np.random.uniform(E_RANGE[0], E_RANGE[1])

    # iteration variable @t is over "next obs" indices
    # for t in range(traj_len):
    for idx, t in enumerate(index):
        obs = deepcopy(env.reset_to({"states": states[t]}))

        # extract datagen info
        if add_datagen_info:
            datagen_info = env.base_env.get_datagen_info(action=actions[t])
        else:
            datagen_info = {}

        # infer reward signal
        # note: our tasks use reward r(s'), reward AFTER transition, so this is
        #       the reward for the current timestep
        r = env.get_reward()

        # infer done signal
        done = False
        if (done_mode == 1) or (done_mode == 2):
            # done = 1 at end of trajectory
            done = done or (t == traj_len)
        if (done_mode == 0) or (done_mode == 2):
            # done = 1 when s' is task success state
            done = done or env.is_success()["task"]
        done = int(done)

        # get the absolute action
        action_abs = env.base_env.convert_rel_to_abs_action(actions[t])

        # get the camera info
        cam_info = {}

        sim = env.env.sim
        camera_name = 'robot0_activeview'          # 选择要复用的相机槽位
        cam_id = sim.model.camera_name2id(camera_name)

        eef_pos = obs['robot0_base_to_eef_pos']
        extrinsic_matrix_newcam = compose_camera_extrinsic(lookat=eef_pos, distance=distance, azimuth=azimuth, elevation=elevation)
        camera_pos_new, camera_rot_new = invert_camera_extrinsic_matrix(extrinsic_matrix_newcam)

        set_camera_pose_world(sim, cam_id, camera_pos_new, camera_rot_new)

        obs = deepcopy(env.reset_to({"states": states[t]}))

        for i, name in enumerate(env.env.camera_names):
            cam_info[name + "_intrinsics"] = get_camera_intrinsic_matrix(env.env.sim, name, env.env.camera_heights[i], env.env.camera_widths[i])
            cam_info[name + "_extrinsics"] = get_camera_extrinsic_matrix(env.env.sim, name)

        # collect transition
        traj["states"].append(states[t])
        traj["actions"].append(actions[t])
        traj["obs"].append(obs)
        traj["rewards"].append(r)
        traj["dones"].append(done)
        traj["datagen_info"].append(datagen_info)
        traj["actions_abs"].append(action_abs)
        traj["cam_info"].append(cam_info)

    # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    traj["datagen_info"] = TensorUtils.list_of_flat_dict_to_dict_of_list(
        traj["datagen_info"]
    )
    traj["cam_info"] = TensorUtils.list_of_flat_dict_to_dict_of_list(
        traj["cam_info"]
    )

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return traj


""" The process that writes over the generated files to memory """


def write_traj_to_file(
    args, output_path, total_samples, total_run, processes, mul_queue
):
    f = h5py.File(args.dataset, "r")
    f_out = h5py.File(output_path, "w")
    data_grp = f_out.create_group("data")
    start_time = time.time()
    num_processed = 0
    
    # 新增：记录有多少个工作进程已经发送了结束信号
    finished_processes_count = 0

    try:
        # 改为死循环，依靠接收到的 None 信号来退出
        while True:
            # 使用阻塞式获取，如果队列为空会等待，直到有数据或收到 None
            item = mul_queue.get()

            # --- 哨兵逻辑开始 ---
            if item is None:
                finished_processes_count += 1
                # 如果收到的结束信号数量等于启动的进程数，说明所有人都干完活了且数据都发完了
                if finished_processes_count == processes:
                    break
                # 否则继续等待其他进程
                continue
            # --- 哨兵逻辑结束 ---

            # 如果不是 None，则是正常数据
            num_processed = num_processed + 1
            ep = item[0]
            traj = item[1]
            process_num = item[2]
            
            try:
                ep_data_grp = data_grp.create_group(ep)
                ep_data_grp.create_dataset(
                    "actions", data=np.array(traj["actions"])
                )
                ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
                ep_data_grp.create_dataset(
                    "rewards", data=np.array(traj["rewards"])
                )
                ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
                ep_data_grp.create_dataset(
                    "actions_abs", data=np.array(traj["actions_abs"])
                )
                for k in traj["obs"]:
                    if args.no_compress:
                        ep_data_grp.create_dataset(
                            "obs/{}".format(k), data=np.array(traj["obs"][k])
                        )
                    else:
                        ep_data_grp.create_dataset(
                            "obs/{}".format(k),
                            data=np.array(traj["obs"][k]),
                            compression="gzip",
                        )
                    if args.include_next_obs:
                        if args.no_compress:
                            ep_data_grp.create_dataset(
                                "next_obs/{}".format(k),
                                data=np.array(traj["next_obs"][k]),
                            )
                        else:
                            ep_data_grp.create_dataset(
                                "next_obs/{}".format(k),
                                data=np.array(traj["next_obs"][k]),
                                compression="gzip",
                            )

                if "datagen_info" in traj:
                    for k in traj["datagen_info"]:
                        ep_data_grp.create_dataset(
                            "datagen_info/{}".format(k),
                            data=np.array(traj["datagen_info"][k]),
                        )

                if "cam_info" in traj:
                    for k in traj["cam_info"]:
                        ep_data_grp.create_dataset(
                            "cam_info/{}".format(k),
                            data=np.array(traj["cam_info"][k]),
                        )

                # copy action dict (if applicable)
                if "data/{}/action_dict".format(ep) in f:
                    action_dict = f["data/{}/action_dict".format(ep)]
                    for k in action_dict:
                        ep_data_grp.create_dataset(
                            "action_dict/{}".format(k),
                            data=np.array(action_dict[k][()]),
                        )

                # episode metadata
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"][
                    "model"
                ]  # model xml for this episode
                ep_data_grp.attrs["ep_meta"] = traj["initial_state_dict"][
                    "ep_meta"
                ]  # ep meta data for this episode
                # if "ep_meta" in f["data/{}".format(ep)].attrs:
                #     ep_data_grp.attrs["ep_meta"] = f["data/{}".format(ep)].attrs["ep_meta"]
                ep_data_grp.attrs["num_samples"] = traj["actions"].shape[
                    0
                ]  # number of transitions in this episode

                total_samples.value += traj["actions"].shape[0]
            except Exception as e:
                print("++" * 50)
                print(
                    f"Error at Process {process_num} on episode {ep} with \n\n {e}"
                )
                print("++" * 50)
                raise Exception("Write out to file has failed")
            
            # 注意：这里的 total_run.value 可能不会实时更新，因为我们现在依赖 finished_processes_count
            # 如果仅仅是为了日志打印，可以用 finished_processes_count 替代 total_run.value
            print(
                "ep {}: wrote {} transitions to group {} at process {} with {} finished. Datagen rate: {:.2f} sec/demo".format(
                    num_processed,
                    ep_data_grp.attrs["num_samples"],
                    ep,
                    process_num,
                    finished_processes_count, 
                    (time.time() - start_time) / num_processed,
                )
            )
    except KeyboardInterrupt:
        print("Control C pressed. Closing File and ending \n\n\n\n\n\n\n")

    if "mask" in f:
        f.copy("mask", f_out)

    # global metadata
    data_grp.attrs["total"] = total_samples.value
    env_meta = DatasetUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    if args.generative_textures:
        env_meta["env_kwargs"]["generative_textures"] = "100p"
    if args.randomize_cameras:
        env_meta["env_kwargs"]["randomize_cameras"] = True
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=args.camera_names,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        reward_shaping=args.shaped,
    )
    print("total processes end {}".format(finished_processes_count))
    data_grp.attrs["env_args"] = json.dumps(
        env.serialize(), indent=4
    )  # environment info
    print("Wrote {} total samples to {}".format(total_samples.value, output_path))

    f_out.close()
    f.close()

    DatasetUtils.extract_action_dict(dataset=output_path)
    DatasetUtils.make_demo_ids_contiguous(dataset=output_path)
    # ... (Filter logic stays the same) ...
    for num_demos in [10, 20, 30, 40, 50, 100, 200, 300, 500, 1000, 2000, 5000, 10000]:
         DatasetUtils.filter_dataset_size(output_path, num_demos=num_demos)

    print("Writing has finished")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    return


# runs multiple trajectory. If there has been an unrecoverable error, the system puts the current work back into the queue and exits
def extract_multiple_trajectories(
    process_num, current_work_array, work_queue, lock, args2, num_finished, mul_queue
):
    try:
        extract_multiple_trajectories_with_error(
            process_num, current_work_array, work_queue, lock, args2, mul_queue
        )
    except Exception as e:
        work_queue.put(current_work_array[process_num])
        print("*>*" * 50)
        print("Error process num {}:".format(process_num))
        print(e)
        print(traceback.format_exc())
        print("*>*" * 50)
        print()

    # num_finished.value = num_finished.value + 1
    mul_queue.put(None)


def retrieve_new_index(process_num, current_work_array, work_queue, lock):
    """
    可靠地获取下一个工作索引：
    之前使用 work_queue.empty() 会出现竞态导致最后一个任务丢失。
    现在改为直接 try get_nowait() 捕获 queue.Empty。
    """
    with lock:
        try:
            tmp = work_queue.get_nowait()
        except queue.Empty:
            return -1
        current_work_array[process_num] = tmp
        return tmp


def extract_multiple_trajectories_with_error(
    process_num, current_work_array, work_queue, lock, args, mul_queue
):
    # create environment to use for data processing

    if args.add_datagen_info:
        import mimicgen.utils.file_utils as MG_FileUtils

        env_meta = MG_FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    else:
        env_meta = DatasetUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    if args.generative_textures:
        env_meta["env_kwargs"]["generative_textures"] = "100p"
    if args.randomize_cameras:
        env_meta["env_kwargs"]["randomize_cameras"] = True
    else:
        env_meta["env_kwargs"]["randomize_cameras"] = False
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=args.camera_names,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        reward_shaping=args.shaped,
    )

    start_time = time.time()

    print("==== Using environment with the following metadata ====")
    print(json.dumps(env.serialize(), indent=4))
    print("")

    # list of all demonstration episodes (sorted in increasing number order)
    f = h5py.File(args.dataset, "r")
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [
            elem.decode("utf-8")
            for elem in np.array(f["mask/{}".format(args.filter_key)])
        ]
    else:
        demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[: args.n]

    ind = retrieve_new_index(process_num, current_work_array, work_queue, lock)
    # while (not work_queue.empty()) and (ind != -1):
    while ind != -1:
        try:
            # print("Running {} index".format(ind))
            ep = demos[ind]

            # prepare initial state to reload from
            states = f["data/{}/states".format(ep)][()]
            initial_state = dict(states=states[0])
            initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
            initial_state["ep_meta"] = f["data/{}".format(ep)].attrs.get(
                "ep_meta", None
            )

            # extract obs, rewards, dones
            actions = f["data/{}/actions".format(ep)][()]

            traj = extract_trajectory(
                env=env,
                initial_state=initial_state,
                states=states,
                actions=actions,
                done_mode=args.done_mode,
                add_datagen_info=args.add_datagen_info,
                interval=args.interval
            )

            # maybe copy reward or done signal from source file
            if args.copy_rewards:
                traj["rewards"] = f["data/{}/rewards".format(ep)][()]
            if args.copy_dones:
                traj["dones"] = f["data/{}/dones".format(ep)][()]

            ep_grp = f["data/{}".format(ep)]

            states = ep_grp["states"][()]
            initial_state = dict(states=states[0])
            initial_state["model"] = ep_grp.attrs["model_file"]
            initial_state["ep_meta"] = ep_grp.attrs.get("ep_meta", None)

            # store transitions

            # IMPORTANT: keep name of group the same as source file, to make sure that filter keys are
            #            consistent as well
            # print("(process {}): ADD TO QUEUE index {}".format(process_num, ind))
            mul_queue.put([ep, traj, process_num])

            ind = retrieve_new_index(process_num, current_work_array, work_queue, lock)
        except Exception as e:
            print("_" * 50)
            print("Process {}:".format(process_num))
            print("Error processing demo index {}: {}".format(ind, e))
            print(traceback.format_exc())
            print("_" * 50)
            del env
            env = EnvUtils.create_env_for_data_processing(  # when it errors, it like blows up the environment for some reason
                env_meta=env_meta,
                camera_names=args.camera_names,
                camera_height=args.camera_height,
                camera_width=args.camera_width,
                reward_shaping=args.shaped,
            )

    f.close()
    print("Process {} finished".format(process_num))


def dataset_states_to_obs_multiprocessing(args):
    # create environment to use for data processing

    # output file in same directory as input file
    output_name = args.output_name
    if output_name is None:
        if len(args.camera_names) == 0:
            output_name = os.path.basename(args.dataset)[:-5] + "_ld.hdf5"
        else:
            image_suffix = str(args.camera_width)
            image_suffix = (
                image_suffix + "_randcams" if args.randomize_cameras else image_suffix
            )
            if args.generative_textures:
                output_name = os.path.basename(args.dataset)[
                    :-5
                ] + "_gentex_im{}.hdf5".format(image_suffix)
            else:
                output_name = os.path.basename(args.dataset)[:-5] + "_im{}.hdf5".format(
                    image_suffix
                )

    output_path = os.path.join(os.path.dirname(args.dataset), output_name)

    print("input file: {}".format(args.dataset))
    print("output file: {}".format(output_path))

    f = h5py.File(args.dataset, "r")
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [
            elem.decode("utf-8")
            for elem in np.array(f["mask/{}".format(args.filter_key)])
        ]
    else:
        demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    if args.n is not None:
        demos = demos[: args.n]

    num_demos = len(demos)
    f.close()

    env_meta = DatasetUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    num_processes = args.num_procs

    index = multiprocessing.Value("i", 0)
    lock = multiprocessing.Lock()
    total_samples_shared = multiprocessing.Value("i", 0)
    num_finished = multiprocessing.Value("i", 0)
    mul_queue = multiprocessing.Queue()
    work_queue = multiprocessing.Queue()
    for idx in range(num_demos):
        work_queue.put(idx)
    current_work_array = multiprocessing.Array("i", num_processes)
    processes = []
    for i in range(num_processes):
        process = multiprocessing.Process(
            target=extract_multiple_trajectories,
            args=(
                i,
                current_work_array,
                work_queue,
                lock,
                args,
                num_finished,
                mul_queue,
            ),
        )
        processes.append(process)

    process1 = multiprocessing.Process(
        target=write_traj_to_file,
        args=(
            args,
            output_path,
            total_samples_shared,
            num_finished,
            num_processes,
            mul_queue,
        ),
    )
    processes.append(process1)

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    print("Finished Multiprocessing")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to input hdf5 dataset",
    )
    # name of hdf5 to write - it will be in the same directory as @dataset
    parser.add_argument(
        "--output_name",
        type=str,
        help="name of output hdf5 dataset",
    )

    parser.add_argument(
        "--filter_key",
        type=str,
        help="filter key for input dataset",
    )

    # specify number of demos to process - useful for debugging conversion with a handful
    # of trajectories
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are processed",
    )

    # flag for reward shaping
    parser.add_argument(
        "--shaped",
        action="store_true",
        help="(optional) use shaped rewards",
    )

    # camera names to use for observations
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs="+",
        default=[
            # "robot0_agentview_left",
            # "robot0_agentview_right",
            "robot0_eye_in_hand",
            # "robot0_handview_left",
            # "robot0_handview_right",
            # "robot0_handview_front",
            "robot0_agentview_center",
            # "robot0_frontview",
            # "robot0_birdview",
            "robot0_activeview",
        ],
        help="(optional) camera name(s) to use for image observations. Leave out to not use image observations.",
    )

    parser.add_argument(
        "--camera_height",
        type=int,
        default=128,
        help="(optional) height of image observations",
    )

    parser.add_argument(
        "--camera_width",
        type=int,
        default=128,
        help="(optional) width of image observations",
    )

    # specifies how the "done" signal is written. If "0", then the "done" signal is 1 wherever
    # the transition (s, a, s') has s' in a task completion state. If "1", the "done" signal
    # is one at the end of every trajectory. If "2", the "done" signal is 1 at task completion
    # states for successful trajectories and 1 at the end of all trajectories.
    parser.add_argument(
        "--done_mode",
        type=int,
        default=0,
        help="how to write done signal. If 0, done is 1 whenever s' is a success state.\
            If 1, done is 1 at the end of each trajectory. If 2, both.",
    )

    # flag for copying rewards from source file instead of re-writing them
    parser.add_argument(
        "--copy_rewards",
        action="store_true",
        help="(optional) copy rewards from source file instead of inferring them",
    )

    # flag for copying dones from source file instead of re-writing them
    parser.add_argument(
        "--copy_dones",
        action="store_true",
        help="(optional) copy dones from source file instead of inferring them",
    )

    # flag to include next obs in dataset
    parser.add_argument(
        "--include-next-obs",
        action="store_true",
        help="(optional) include next obs in dataset",
    )

    # flag to disable compressing observations with gzip option in hdf5
    parser.add_argument(
        "--no_compress",
        action="store_true",
        help="(optional) disable compressing observations with gzip option in hdf5",
    )

    parser.add_argument(
        "--num_procs",
        type=int,
        default=5,
        help="number of parallel processes for extracting image obs",
    )

    parser.add_argument(
        "--add_datagen_info",
        action="store_true",
        help="(optional) add datagen info (used for mimicgen)",
    )

    parser.add_argument("--generative_textures", action="store_true")

    parser.add_argument("--randomize_cameras", action="store_true")

    parser.add_argument("--interval",
        type=int,
        default=1,
        help="interval of stored state",
    )

    args = parser.parse_args()
    dataset_states_to_obs_multiprocessing(args)
