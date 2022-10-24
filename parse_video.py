import argparse
import cv2
import os
import glob
from pathlib import Path

def vid_to_frames(fname):
    fname_base = Path(fname).stem
    try:
       os.mkdir(fname_base)
    except FileExistsError as error:
        pass
    vidcap = cv2.VideoCapture(fname)
    success,image = vidcap.read()
    count = 0
    while success:
        frame_fname = fname_base + '/frame_{:04d}.jpg'.format(count)
        cv2.imwrite(frame_fname, image)
        success,image = vidcap.read()
        count += 1

def convert_vid_folder(folder):
    os.chdir(folder)
    files = glob.glob('*.mp4')
    count = 0
    for file in files:
        vid_to_frames(file)
        if (count % 10 == 0):
            print('{}/{}'.format(count, len(files)))
        count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str)
    args = parser.parse_args()
    convert_vid_folder(args.folder)