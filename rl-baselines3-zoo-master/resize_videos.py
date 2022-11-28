# Import required modules
import numpy as np
import imageio
import os
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import skvideo.io
import fnmatch

'''

cd /home/codysoccerman/Documents/classes/Fall_22/Deep_Learning/Project/rl-baselines3-zoo-master
conda activate mujoco_py
python3 resize_videos.py

'''


env_name = "PandaPickAndPlace-v1"
recording_path = "recording_data/" + env_name

# Cropped image
crop_size = 208
width = 255
height = 255
print(int(width/2-crop_size/2), int(width/2+crop_size/2), int(height/2-crop_size/2), int(height/2+crop_size/2))

start_path = recording_path + "/videos/"
final_path = recording_path + "/videos_cropped/"

# low res 
low_res = 128 #96
final_path_low_res = recording_path + "/videos_low_res/"

# grayscaled
low_res = 128 #96
final_path_grayscaled = recording_path + "/videos_grayscaled/"

# video read

#video = skvideo.io.vread(final_path_grayscaled + "np_video_3.mp4")
#video = np.asarray(video, dtype='uint8')
#video_new = video[:,:,:,1]

original_count = len(fnmatch.filter(os.listdir(start_path), '*.*'))
print('File Count:', original_count)
los_res_count = len(fnmatch.filter(os.listdir(final_path_low_res), '*.*'))
print('File Count:', los_res_count)

count = 1
if los_res_count > 0:
    for filenumber in range(los_res_count-1, original_count+1):
        #for filenumber in os.listdir(count):
        #if filename.endswith(".mp4"):
        filename = "video_" + str(filenumber) + ".mp4"
        print(filenumber)

        # Load video
        video = skvideo.io.vread(start_path + filename)
        image_array = np.asarray(video, dtype='uint8')

        #image_array = np.load(start_path + filename)

        filename_base = os.path.splitext(filename)[0]
        #video_writer = imageio.get_writer(final_path + filename_base + ".mp4", fps=20)
        video_writer2 = imageio.get_writer(final_path_low_res + filename_base + ".mp4", fps=20)
        video_writer3 = imageio.get_writer(final_path_grayscaled + filename_base + ".mp4", fps=20)

        for single_image in image_array:
            # crop image
            #single_image = single_image[int(height/2-crop_size/2) : int(height/2+crop_size/2), int(width/2-crop_size/2) : int(width/2+crop_size/2)]
            #video_writer.append_data(np.asarray(single_image))

            # save to videos_low_res
            low_res_image = np.array( Image.fromarray(single_image).resize((low_res,low_res))) #96x96X3
            video_writer2.append_data(np.asarray(low_res_image))

            # save to videos_low_res
            grayscaled_image = Image.fromarray(low_res_image).convert('L') #96x96x1
            video_writer3.append_data(np.asarray(grayscaled_image))
        
        #video_writer.close()
        video_writer2.close()
        video_writer3.close()
        count = count + 1




if los_res_count == 0:
    for filenumber in range(los_res_count+1, original_count+1):
        #for filenumber in os.listdir(count):
        #if filename.endswith(".mp4"):
        filename = "video_" + str(filenumber) + ".mp4"
        print(filenumber)

        # Load video
        video = skvideo.io.vread(start_path + filename)
        image_array = np.asarray(video, dtype='uint8')

        #image_array = np.load(start_path + filename)

        filename_base = os.path.splitext(filename)[0]
        #video_writer = imageio.get_writer(final_path + filename_base + ".mp4", fps=20)
        video_writer2 = imageio.get_writer(final_path_low_res + filename_base + ".mp4", fps=20)
        video_writer3 = imageio.get_writer(final_path_grayscaled + filename_base + ".mp4", fps=20)

        for single_image in image_array:
            # crop image
            #single_image = single_image[int(height/2-crop_size/2) : int(height/2+crop_size/2), int(width/2-crop_size/2) : int(width/2+crop_size/2)]
            #video_writer.append_data(np.asarray(single_image))

            # save to videos_low_res
            low_res_image = np.array( Image.fromarray(single_image).resize((low_res,low_res))) #96x96X3
            video_writer2.append_data(np.asarray(low_res_image))

            # save to videos_low_res
            grayscaled_image = Image.fromarray(low_res_image).convert('L') #96x96x1
            video_writer3.append_data(np.asarray(grayscaled_image))
        
        #video_writer.close()
        video_writer2.close()
        video_writer3.close()
        count = count + 1

'''
if los_res_count == 0:   
    for filename in os.listdir(start_path):
        if filename.endswith(".mp4"):
            print(count)

            # Load video
            video = skvideo.io.vread(start_path + filename)
            image_array = np.asarray(video, dtype='uint8')

            #image_array = np.load(start_path + filename)

            filename_base = os.path.splitext(filename)[0]
            #video_writer = imageio.get_writer(final_path + filename_base + ".mp4", fps=20)
            video_writer2 = imageio.get_writer(final_path_low_res + filename_base + ".mp4", fps=20)
            video_writer3 = imageio.get_writer(final_path_grayscaled + filename_base + ".mp4", fps=20)
    
            for single_image in image_array:
                # crop image
                #single_image = single_image[int(height/2-crop_size/2) : int(height/2+crop_size/2), int(width/2-crop_size/2) : int(width/2+crop_size/2)]
                #video_writer.append_data(np.asarray(single_image))

                # save to videos_low_res
                low_res_image = np.array( Image.fromarray(single_image).resize((low_res,low_res))) #96x96X3
                video_writer2.append_data(np.asarray(low_res_image))
    
                # save to videos_low_res
                grayscaled_image = Image.fromarray(low_res_image).convert('L') #96x96x1
                video_writer3.append_data(np.asarray(grayscaled_image))
            
            #video_writer.close()
            video_writer2.close()
            video_writer3.close()
            count = count + 1
        else:
            continue
'''

# ckeck


#plt.imshow(single_image, interpolation='nearest')
#plt.show()
##############################

# https://stackoverflow.com/questions/68398729/how-to-cut-videos-automatically-using-python-with-ffmpeg
# https://www.bogotobogo.com/FFMpeg/ffmpeg_cropping_video_image.php






if False == True:

    import os
    import ffmpeg


    env_name = "PandaPickAndPlace-v1"
    recording_path = "recording_data/" + env_name

    #path = r'C:\Users\user\Desktop\folder'
    start_path = recording_path + "/videos/"
    final_path = recording_path + "/videos_resized_small/"

    for filename in os.listdir(start_path):
        if filename.endswith(".mp4"): 
            command = 'ffmpeg -i ' + os.path.join(start_path, filename) + ' -vf "crop=200:200" ' + os.path.join(final_path, filename) + ' -loglevel quiet'
            os.system(command)
        else:
            continue



    start_path = recording_path + "/videos/"
    final_path = recording_path + "/videos_resized_small/"
    filename = 'video_1.mp4'
    command = 'ffmpeg -i ' + os.path.join(start_path, filename) + ' -vf "crop=200:200" ' + os.path.join(final_path, filename) + ' -loglevel quiet'
    os.system(command)


    start_path = recording_path + "/videos_resized_small/"
    final_path = recording_path + "/videos_low_res/"

    filename = 'video_1.mp4'
    command = 'ffmpeg -i ' + os.path.join(start_path, filename) + ' -vf scale=100:-1 ' + os.path.join(final_path, filename) + ' -loglevel quiet'
    print(command)
    os.system(command)

##############################################################################

