import cv2
import sys

frames_list = sys.argv[1]
original_video_path = sys.argv[2]
output_video_path = sys.argv[3]

width = int(sys.argv[4])
height = int(sys.argv[5])

with open(frames_list, 'r') as f:
    frames = f.readlines()

og_vid = cv2.VideoCapture(original_video_path)
selected_vid = cv2.VideoWriter()
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
selected_vid.open(output_video_path, fourcc, 30, (width, height))

if not selected_vid.isOpened():
    print("Error opening output video")
    sys.exit(1)

#for frame_num in frames:
#    og_vid.set(cv2.CAP_PROP_POS_FRAMES, int(frame_num))
    ret, frame = og_vid.read()
#    selected_vid.write(frame)

selected_vid.release()
