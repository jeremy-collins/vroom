"""
A convenience script to playback random demonstrations from
a set of demonstrations stored in a hdf5 file.

Arguments:
    --folder (str): Path to demonstrations
    --use-actions (optional): If this flag is provided, the actions are played back
        through the MuJoCo simulator, instead of loading the simulator states
        one by one.
    --visualize-gripper (optional): If set, will visualize the gripper site

Example:
    $ python playback_demonstrations_from_hdf5.py --folder ../models/assets/demonstrations/SawyerPickPlace/
"""

import argparse
import json
import os
import random
import imageio


import h5py
import numpy as np
import time

import robosuite
from robosuite.utils.mjcf_utils import postprocess_model_xml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        help="Path to your demonstration folder that contains the demo.hdf5 file, e.g.: "
        "'path_to_assets_dir/demonstrations/YOUR_DEMONSTRATION'",
    ),
    parser.add_argument(
        "--use-actions",
        action="store_true",
    )
    args = parser.parse_args()

    demo_path = args.folder
    hdf5_path = os.path.join(demo_path, "demo.hdf5")
    video_path = os.path.join(demo_path, 'videos')
    try:
        os.mkdir(video_path)
    except FileExistsError as error:
        pass
    f = h5py.File(hdf5_path, "r")
    env_name = f["data"].attrs["env"]
    # env_info = json.loads(f["data"].attrs["env"])
    env_info = f['data'].attrs['env']

    env = robosuite.make(
        env_info,
        has_renderer=True,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        use_object_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    # list of all demonstrations episodes
    demos = list(f["data"].keys())

    for ep in demos:
        # print("Playing back random episode... (press ESC to quit)")
        print('Playing back episode {}... (press ESC to quit)'.format(ep))

        # # select an episode randomly
        # ep = random.choice(demos)

        # read the model xml, using the metadata stored in the attribute for this episode
        model_xml = f["data/{}".format(ep)].attrs["model_file"]
        with open(os.path.join(demo_path, 'models', model_xml)) as xml_file:
            model_xml_str = xml_file.read()

        env.reset()
        xml = postprocess_model_xml(model_xml_str)
        env.reset_from_xml_string(xml)
        env.sim.reset()
        # env.viewer.set_camera(0)

        # load the flattened mujoco states
        states = f["data/{}/states".format(ep)][()]

        if args.use_actions:

            # load the initial state
            env.sim.set_state_from_flattened(states[0])
            env.sim.forward()

            # load the actions and play them back open-loop
            # actions = np.array(f["data/{}/actions".format(ep)][()])
            # concatenate joint velocities and gripper actuations to form action space
            joint_vel = np.array(f["data/{}/joint_velocities".format(ep)][()])
            gripper_act = np.array(f["data/{}/gripper_actuations".format(ep)][()])
            actions = np.concatenate((joint_vel, gripper_act), axis=1)
            num_actions = actions.shape[0]

            for j, action in enumerate(actions):
                env.step(action)
                env.render()

                if j < num_actions - 1:
                    # ensure that the actions deterministically lead to the same recorded states
                    state_playback = env.sim.get_state().flatten()
                    if not np.all(np.equal(states[j + 1], state_playback)):
                        err = np.linalg.norm(states[j + 1] - state_playback)
                        print(f"[warning] playback diverged by {err:.2f} for ep {ep} at step {j}")

        else:
            video_writer = imageio.get_writer(os.path.join(video_path, "demo_{}.mp4".format(ep)), fps=120)
            frame_count = 1
            fps_time_beg = time.time()

            # force the sequence of internal mujoco states one by one
            for state in states:
                env.sim.set_state_from_flattened(state)
                env.sim.forward()
                video_img = np.array(env.sim.render(height=256, width=256, camera_name='frontview')[::-1])
                video_writer.append_data(video_img)
                # env.render()
                if (time.time() - fps_time_beg >= 1.0):
                    print('fps: {}'.format(frame_count))
                    fps_time_beg = time.time()
                    frame_count = 1
                else:
                    frame_count += 1

            video_writer.close()


    f.close()
