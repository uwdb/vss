import numpy as np
import cv2

def project(keypoints1, keypoints2, matches):
    source_points = np.float32([keypoints1[m.queryIdx] for m in matches]).reshape(-1, 1, 2)
    destination_points = np.float32([keypoints2[m.trainIdx] for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(source_points, destination_points, cv2.RANSAC, 3)
    Hi = np.linalg.inv(H)

    return H / H[2,2], Hi / Hi[2,2]

