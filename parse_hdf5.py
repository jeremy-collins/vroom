import argparse
import os
import glob
from pathlib import Path

import h5py
import numpy as np

# example usage: python parse_hdf5.py --folder RoboTurkPilot/bins-Bread/

def separate_joint_csv(demo_path):
    data_path = os.path.join(demo_path, 'jointdata')

    os.chdir(data_path)
    files = glob.glob('*.npy')
    for file in files:
        fname_base = Path(file).stem
        try:
            os.mkdir(fname_base)
        except FileExistsError as error:
            pass

        dat = np.load(file)
        for i in range(0, dat.shape[0]):
            np.save(os.path.join(fname_base, 'frame_{:04d}.npy'.format(i)))


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
    parser.add_argument(
        '--separate',
        action='store_true',
    )

    args = parser.parse_args()

    demo_path = args.folder

    if (args.separate):
        separate_joint_csv(demo_path)
    else:
        get_joint_csv(demo_path)
