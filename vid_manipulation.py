import cv2
import numpy as np
import os
import glob



def create_frames(video_path, images_path):
    vid = cv2.VideoCapture(video_path)

    frame_curr = 0
    while(True):
        ret, frame = vid.read()

        name = images_path + '/' + str(frame_curr) + '.jpg'
        print("Creating " + name)
        cv2.imwrite(name, frame)

        frame_curr += 1
        if frame_curr >= 400:
            break
    vid.release()
    return frame_curr, images_path



def create_vid(num_frames, video_path, images_path):
    img_array = []
    fnames = []
    for i in range(0, num_frames):
        fnames.append(i)

    fnames = sorted(fnames)
    for file in fnames:
        file = images_path + '/' + str(file) + '.jpg'
        print(file)
        img = cv2.imread(file)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img) 
    
    out = cv2.VideoWriter(video_path,cv2.VideoWriter_fourcc(*'mp4v'), 20.0, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()