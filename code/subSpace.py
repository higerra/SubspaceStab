import subprocess
import cv2 as cv
import os
from os.path import join
import argparse


# Pass command line inputs for the stabilization procedure
parser = argparse.ArgumentParser()
# Add input file path, default type is string
parser.add_argument("-i", action="store", dest="file_in")
# Add output file path, default type is string
parser.add_argument("-o", action="store", dest="file_out")
# Whether to crop video
parser.add_argument("--crop", action="store_true", dest="bool_crop")
# read cmd line arguments
args_read = parser.parse_args()
# File path
file_path = args_read.file_in
out_path = args_read.file_out
bool_crop = args_read.bool_crop

fourcc_avi = cv.VideoWriter_fourcc(*'XVID')
fourcc_mp4 = cv.VideoWriter_fourcc(*'mp4v')
cap = cv.VideoCapture(file_path)
# Set output FPS rate
fps = int(cap.get(cv.CAP_PROP_FPS))
cap.release()
# Temporary location for *.jpg files cleared later
image_folder = 'build'


# Extract input file name sans extension and vid type of either mp4 or avi
[in_name, vid_type] = file_path.split('/')[-1].split('.')
# Define the codec for output video
if vid_type == 'mp4':
    fourcc = fourcc_mp4
    out_name = in_name + "_avg_stab_out.mp4"
elif vid_type == 'avi':
    fourcc = fourcc_avi
    out_name = in_name + "_avg_stab_out.avi"
else:
    print("Unsupported video file type")
    exit(-1)
# print("Currently Processing folder {0}, file {1}, type {2}".format(file_path, in_name, vid_type))
# Try-Except block in case processing a video fails (due to failure of feature extraction etc.)
# Invoke the executable on the current input video, also specify intermediate output files location
subprocess.check_call(['./build/SubspaceStab', file_path, '--output=./build/result_{0}'.format(in_name),
                       '--resize=false', '--crop={0}'.format(bool_crop)])
# create an ordered list of .jpg images to stitch together
images = [img for img in os.listdir(image_folder) if (img.endswith(".jpg") and not img.startswith("result_test"))]
images.sort()
# Read one frame to get started
frame = cv.imread(join(image_folder, images[0]))
height, width, layers = frame.shape
# Set up output video stream
out = cv.VideoWriter(out_path, fourcc, fps, frameSize=(width, height))
for image in images:
    # print("Processing Image {0}".format(image))
    out.write(cv.imread(os.path.join(image_folder, image)))
# delete intermediate *.jpg files
subprocess.call('rm build/*.jpg', shell=True)
out.release()
