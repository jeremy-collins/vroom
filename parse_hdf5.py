import argparse
import os


import h5py
import numpy as np

# example usage: python parse_hdf5.py --folder RoboTurkPilot/bins-Bread/

def get_joint_csv(demo_path):
    hdf5_path = os.path.join(demo_path, "demo.hdf5")
    video_path = os.path.join(demo_path, 'videos')
    data_path = os.path.join(demo_path, 'jointdata')

    try:
        os.mkdir(data_path)
    except FileExistsError as error:
        pass

    f = h5py.File(hdf5_path, "r")

    demos = list(f["data"].keys())

    for i in range(0, len(demos)):
        ep = demos[i]
        states = f["data/{}/states".format(ep)][()]
        joint_vel = np.array(f["data/{}/joint_velocities".format(ep)][()])
        gripper_act = np.array(f["data/{}/gripper_actuations".format(ep)][()])
        actions = np.concatenate((joint_vel, gripper_act), axis=1)

        np.save(os.path.join(data_path, '{}_jointdata.npy'.format(ep)), actions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str
    )

    args = parser.parse_args()

    demo_path = args.folder

    get_joint_csv(demo_path)