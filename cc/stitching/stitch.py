import math
import numpy as np
import cv2


PIXEL_MAX = 255.0
THRESHOLD = 0.7 # Lowe et al.
MINIMUM_MATCHES_REQUIRED = 6

FLANN_INDEX_KDTREE = 0
FLANN_INDEX_LSH = 6

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return "+inf"
    else:
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


image1 = cv2.imread('p0/001.png')
image2 = cv2.imread('p30/001.png')

image1 = cv2.copyMakeBorder(image1, image1.shape[0]/2, image1.shape[0]/2, image1.shape[1]/2, image1.shape[1]/2, cv2.BORDER_CONSTANT, 0)

orb = cv2.ORB_create()
# Need to do this just on grayscale
keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

index_parameters = dict(algorithm = FLANN_INDEX_LSH, table_number = 6, key_size = 12, multi_probe_level = 1)
search_parameters = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_parameters, search_parameters)
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# ratio test as per Lowe's paper
matches = [(m, n) for m, n in matches if m.distance < THRESHOLD * n.distance]

if len(matches) < MINIMUM_MATCHES_REQUIRED:
    print('Insufficient matches, should encode separately')

print(len(matches))

#img3 = cv2.drawMatchesKnn(image1, keypoints1, image2, keypoints2, matches, None) #,**draw_params)
#cv2.imwrite('out3.png', img3)

source_points = np.float32([keypoints1[m.queryIdx].pt for m, n in matches]).reshape(-1,1,2)
destination_points = np.float32([keypoints2[m.trainIdx].pt for m, n in matches]).reshape(-1,1,2)

H, mask = cv2.findHomography(source_points, destination_points, cv2.RANSAC, 3)
Hi = np.linalg.inv(H)

Hx = [
 [3.52775073,
  -0.0747832656,
  -4851.04395],
 [0.71989572,
  2.5550611,
  -1703.91907],
 [0.000670183275,
  -2.97702718e-05,
  1]]

#H = np.array(Hx)

#    matchesMask = mask.ravel().tolist()

print H
print '---'
print np.linalg.inv(H)
print '---'
#print np.linalg.inv(H).dot(H)
#print '---'

# Left overlap at x0
i = np.array([0, 0, 1])
Hi = np.linalg.inv(H)
H0 = Hi.dot(i)
H0n = H0 / H0[2]
x0 = H0n[0]
p0 = int(x0)

p0 = int(Hi[0, 2] / Hi[2, 2])

# Right of overlap at x1

right = image1.shape[1] * 3/4
j = np.array([right, 0, 1])
H1 = H.dot(j)
print 'H1 %s' % H1
H1n = H1 / H1[2]
x1 = H1n[0]
p1 = int(x1)


# Bottom-left corner of of top/left overlap triangle at y0
y0 = int(Hi[1, 2] / Hi[2, 2])
print y0

print p0, right, p1
#print 'p1: %d' % p1
#print M
#a = np.eye(3)
#a[0][2] = 6000
#a[1][2] = 1000
#M = M.dot(a)
#print M

#cv2.imwrite('out1.png', image1)

warped = cv2.warpPerspective(image2, Hi, tuple(reversed(image1.shape[:2]))) #, (300,300))
output = np.array(image1)

np.copyto(output, warped, where = warped != 0)
#mask = warped != 0
#mask &= output != 0
#output += warped
#output[mask] = (image1 / 2) + (warped / 2)

#out2 += image1

output[:, p0:p0+3, :] = 0
output[:, p0:p0+3, 0] = 255

output[:, right:right+3, :] = 0
output[:, right:right+3, 2] = 255

output[:, right+3:, :] = 0
output[output.shape[0]/4:output.shape[0]*3/4, right+3:right+3+(image2.shape[1]-p1), :] = image2[:, p1:, :]

#cv2.imwrite('out1.png', image1)
#cv2.imwrite('out4.png', output)
#cv2.imwrite('out5.png', warped)

overlap = output[:, p0:right, :]
oleft = image1[:, p0:right, :]
oright = warped[:, p0:right, :]
cv2.imwrite('overlap.png', overlap)
#cv2.imwrite('oleft.png', oleft)
#cv2.imwrite('oright.png', oright)


leftmeanmask = np.repeat((np.sum(oleft, axis=2) == 0)[:,:, np.newaxis], 3, axis=2)
rightmeanmask = np.repeat((np.sum(oright, axis=2) == 0)[:,:, np.newaxis], 3, axis=2)
meanmask = leftmeanmask | rightmeanmask
maskedleft = np.ma.masked_where(leftmeanmask, oleft)
maskedright = np.ma.masked_where(rightmeanmask, oright)


# Left
favorleft = np.array(oright)
np.copyto(favorleft, oleft, where=oleft != 0)
print("Left join PSNR %d vs %d" % (psnr(maskedleft, favorleft), psnr(maskedright, favorleft)))
cv2.imwrite('favorleft.png', favorleft)

# right
favorright = np.array(oleft)
np.copyto(favorright, oright, where=oright != 0)
print("Right join PSNR %d vs %d" % (psnr(maskedleft, favorright), psnr(maskedright, favorright)))
cv2.imwrite('favorright.png', favorright)

# Mean
mean = np.ma.mean([maskedleft, maskedright], axis=0)
print("Mean join PSNR %d vs %d" % (psnr(maskedleft, mean), psnr(maskedright, mean)))
cv2.imwrite('mean.png', mean)

# Interlace
interlace = np.empty((oleft.shape[0] * 2, oleft.shape[1], 3))
interlace[::2, :, :] = oleft
interlace[1::2, :, :] = oright
print("Interlace join PSNR +inf vs +inf")
cv2.imwrite('interlace.png', interlace)

# Delta-Left Interlace (quant=1)
quant = 1
delta = np.empty((oleft.shape[0] * 2, oleft.shape[1], 3))
delta[::2, :, :] = oleft
delta[1::2, :, :] = ((np.array(oright, dtype=np.int32) - oleft) / quant) + 127
recovered = np.array(oleft + ((np.array(delta[1::2, :, :], dtype=np.int32) - 127) * quant), dtype=np.uint8)
print("Delta join PSNR +inf vs %s" % psnr(maskedright, recovered))
cv2.imwrite('delta.png', delta)

# Delta-Left Interlace (quant=4)
quant = 4
delta = np.empty((oleft.shape[0] * 2, oleft.shape[1], 3))
delta[::2, :, :] = oleft
delta[1::2, :, :] = ((np.array(oright, dtype=np.int32) - oleft) / quant) + 127
recovered = np.array(oleft + ((np.array(delta[1::2, :, :], dtype=np.int32) - 127) * quant), dtype=np.uint8)
print("Delta quant=4 PSNR +inf vs %d" % psnr(maskedright, recovered))
cv2.imwrite('delta.png', delta)
