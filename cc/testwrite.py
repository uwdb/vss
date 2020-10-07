import cv2

image = cv2.imread('input.png')
print cv2.imwrite('../mount/wolf/1x12.jpg', image)
