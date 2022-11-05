import argparse
import json
import os
import random
import imageio

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import cv2

import h5py
import numpy as np
import time
import matplotlib.pyplot as plt

import robosuite
from robosuite.utils.mjcf_utils import postprocess_model_xml
from predict_lstm import LSTMPredictor

def sim_environment(data_folder, env, run_id, predictor, save=False, sequence_length=5, frame_size=(64,64)):
    jointdata_folder = os.path.join(data_folder, 'demo_{}_jointdata'.format(run_id))

    video_writer = imageio.get_writer(os.path.join(data_folder, "visualize_demo_{}.mp4".format(run_id)), fps=120)

    past_frames = []
    for i in range(0, sequence_length):
        action = np.load(os.path.join(jointdata_folder, 'frame_{:04d}.npy'.format(i)))
        obs, reward, done, info = env.step(action)
        # env.render()
        render_img = obs['image']

        if (save):
            # frame = cv2.cvtColor(render_img, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(render_img, 0)
            # cv2.imshow('whee', frame)
            # cv2.waitKey(100)
            video_writer.append_data(frame)

        render_img = frame_to_tensor(render_img, frame_size)

        past_frames.append(render_img)

    count = 5
    while (os.path.exists(os.path.join(jointdata_folder, 'frame_{:04d}.npy'.format(count)))):

        frames = torch.stack(past_frames, dim=0)
        frames = frames.detach()
        frames.requires_grad = False
        frames = frames.permute(1, 0, 2, 3)
        frames = frames[None,:]

        action = predictor.predict(frames).numpy()
        obs, reward, done, info = env.step(action[0,:])
        # env.render()
        render_img = obs['image']

        if (save):
            # frame = cv2.cvtColor(render_img, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(render_img, 0)
            # cv2.imshow('whee', frame)
            # cv2.waitKey(100)
            video_writer.append_data(frame)

        render_img = frame_to_tensor(render_img, frame_size)

        past_frames.append(render_img)
        past_frames = past_frames[1:]

        count += 1
        print(count)
        print(action)

    if (save):
        video_writer.close()

def init_environment(demo_path, run_id):
    hdf5_path = os.path.join(demo_path, 'demo.hdf5')
    f = h5py.File(hdf5_path, "r")
    env_info = f['data'].attrs['env']

    env = robosuite.make(
        env_info,
        has_renderer=True,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        camera_depth=True,
        use_object_obs=False,
        reward_shaping=True,
        camera_name="frontview",
        control_freq=100,
    )

    model_xml = f["data/demo_{}".format(run_id)].attrs["model_file"]
    with open(os.path.join(demo_path, 'models', model_xml)) as xml_file:
        model_xml_str = xml_file.read()

    env.reset()
    xml = postprocess_model_xml(model_xml_str)
    env.reset_from_xml_string(xml)
    env.sim.reset()
    # env.viewer.set_camera(0)

    states = f["data/demo_{}/states".format(run_id)][()]
    env.sim.set_state_from_flattened(states[0])
    env.sim.forward()

    return env

def frame_to_tensor(frame, frame_size):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 0)
    frame = cv2.resize(frame, frame_size)
    frame = torch.from_numpy(frame)
    frame = frame.permute(2, 0, 1)
    frame = frame.float()/255.0

    return frame

if __name__ == "__main__":
    frame_size = (64,64)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_folder",
        type=str,
    ),
    parser.add_argument(
        "--demo_path",
        type=str,
    )
    parser.add_argument(
        "--run_id",
        type=int
    )
    parser.add_argument(
        "--index",
        type=str
    )
    parser.add_argument(
        "--save",
        action="store_true"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = LSTMPredictor(num_channels=3, num_kernels=64, kernel_size=(3,3), padding=(1,1), activation="relu", frame_size=frame_size, num_layers=3, index=args.index, device=device)

    env = init_environment(args.demo_path, args.run_id)
    sim_environment(args.data_folder, env, args.run_id, predictor, save=args.save)