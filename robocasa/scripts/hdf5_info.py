import h5py
import json

f = h5py.File("/data/zichen/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToSink/mg/2024-05-04-22-14-06_and_2024-05-07-07-40-17/demo_gentex_im128_randcams.hdf5", "r")
demo = f["data"]["demo_5"]                        # access demo 5
obs = demo["obs"]                                 # obervations across all timesteps
left_img = obs["robot0_agentview_left_image"][:]  # get left camera images in numpy format
ep_meta = json.loads(demo.attrs["ep_meta"])       # get meta data for episode
lang = ep_meta["lang"]                            # get language instruction for episode
print("Instruction:", lang)
f.close()