import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import re
import codecs

def parse_log(fname, dt=False):
    with open(fname, 'r', encoding='cp850') as f:
        lines = f.readlines()

    epochs = []
    epoch = 0
    training_loss = []
    validation_loss = []
    for line in lines:
        if (dt):
            txt = re.split("{+|,+|:+|}+|'+| +", line)
            if ('learning_rate' in line and 'epoch' in line):
                train_loss = float(txt[5])
                epoch = float(txt[17])
                epochs.append(epoch)
                training_loss.append(train_loss)
        else:
            if ("Epoch" in line):
                epoch += 1
                epochs.append(epoch)
            if ("Training loss" in line):
                txt = line.split(' ')
                loss = float(txt[2])
                training_loss.append(loss)
            if ("Validation loss" in line):
                txt = line.split(' ')
                loss = float(txt[2])
                validation_loss.append(loss)

    plt.figure()
    plt.plot(epochs, training_loss, label='Train')
    if (not dt):
        plt.plot(epochs, validation_loss, label='Validation')
    # plt.title('Loss history')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.legend(loc='best')
    logname = os.path.splitext(os.path.basename(fname))[0]
    plt.savefig('log_{}.png'.format(logname), dpi=300)

parser = argparse.ArgumentParser()
parser.add_argument(
    'fname'
)
parser.add_argument(
    '--dt',
    action='store_true'
)

args = parser.parse_args()

parse_log(args.fname, args.dt)