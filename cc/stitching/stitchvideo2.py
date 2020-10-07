import math
import sys

import os
import numpy as np
import cv2


PIXEL_MAX = 255.0
THRESHOLD = 0.7 # Lowe et al.
MINIMUM_MATCHES_REQUIRED = 6

FLANN_INDEX_KDTREE = 0
FLANN_INDEX_LSH = 6


def psnr(img1, img2):
    if img1.shape != img2.shape: return None
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return None
    else:
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def homography(image1, image2):
    orb = cv2.ORB_create()
    # Need to do this just on grayscale
    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

    index_parameters = dict(algorithm = FLANN_INDEX_LSH, table_number = 6, key_size = 12, multi_probe_level = 1)
    search_parameters = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_parameters, search_parameters)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # ratio test as per Lowe's paper
    matches = [m for m in matches if len(m) == 2]
    matches = [(m, n) for m, n in matches if m.distance < THRESHOLD * n.distance]

    if len(matches) < MINIMUM_MATCHES_REQUIRED:
        print('Insufficient matches, should encode separately')

    print(len(matches))

    #img3 = cv2.drawMatchesKnn(image1, keypoints1, image2, keypoints2, matches, None) #,**draw_params)
    #cv2.imwrite('out3.png', img3)

    source_points = np.float32([keypoints1[m.queryIdx].pt for m, n in matches]).reshape(-1,1,2)
    destination_points = np.float32([keypoints2[m.trainIdx].pt for m, n in matches]).reshape(-1,1,2)

    H, mask = cv2.findHomography(source_points, destination_points, cv2.RANSAC, 3)
    return H, np.linalg.inv(H)


def partition(H, Hi, left):
    # Left overlap at x0
    i = np.array([0, 0, 1])
    #Hi = np.linalg.inv(H)
    H0 = Hi.dot(i)
    H0n = H0 / H0[2]
    x0 = H0n[0]
    p0 = int(x0)

    p0 = int(Hi[0, 2] / Hi[2, 2])

    # Right of overlap at x1
    right = left.shape[1] * 3/4
    j = np.array([right, 0, 1])
    H1 = H.dot(j)
    H1n = H1 / H1[2]
    x1 = H1n[0]
    p1 = int(x1)

    ymin = int(H0n[1]) / 2 # Not sure about this
    #print H0n
    #Hy = Hi.dot(np.array([left.shape[1]*3/4, left.shape[0], 1]))
    #Hyn = Hy / Hy[2]
    ymax = left.shape[0]-ymin  # used symmetry, but should calculate explicitly
    #print ymin, ymax, left.shape

    # Bottom-left corner of of top/left overlap triangle at y0
    #y0 = int(Hi[1, 2] / Hi[2, 2])

    return p0, right, p1, (ymin, ymax)


def cut(image1, image2, Hi, (p0, right, p1, (ymin, ymax))):
    warped = cv2.warpPerspective(image2, Hi, tuple(reversed(image1.shape[:2])))
    output = np.array(image1)

    np.copyto(output, warped, where=warped != 0)

    output[:, p0:p0+3, :] = 0
    output[:, p0:p0+3, 0] = 255

    output[:, right:right+3, :] = 0
    output[:, right:right+3, 2] = 255

    output[:, right+3:, :] = 0
    #output[output.shape[0]/4:output.shape[0]*3/4, right+3:right+3+(image2.shape[1]-p1), :] = image2[:, p1:, :]
    #cv2.imwrite('output.png', output)

    overlap = output[:, p0:right, :]
    olwidth = output.shape[1]/2 - overlap.shape[1]
    # No idea why it's image1.shape[1]/4?
    oleft = image1[output.shape[0]/4:output.shape[0]*3/4, image1.shape[1]/4:image1.shape[1]/4+olwidth, :]
    #oleft = image1[output.shape[0]/4:output.shape[0]*3/4, p0:right, :]
    #print ymin, ymax, warped.shape, warped.shape[1] + ymin
    oright = warped[ymin:ymax, p0:right, :]
    unwarpedright = image2[:, p1:, :]

    #return output, overlap, oleft, overlap, unwarpedright #oright
    return oleft, (image1[ymin:ymax, p0:right, :], oright), unwarpedright


def mask(left, right):
    leftmeanmask = np.repeat((np.sum(left, axis=2) == 0)[:,:, np.newaxis], 3, axis=2)
    rightmeanmask = np.repeat((np.sum(right, axis=2) == 0)[:,:, np.newaxis], 3, axis=2)
    mask = leftmeanmask | rightmeanmask
    maskedleft = np.ma.masked_where(leftmeanmask, left)
    maskedright = np.ma.masked_where(rightmeanmask, right)
    return maskedleft, maskedright


def leftjoin(left, right):
    result = np.array(right)
    #result = np.zeros_like(right)
    np.copyto(result, left, where=left != 0)
    #print("Left join PSNR %d vs %d" % (psnr(maskedleft, favorleft), psnr(maskedright, favorleft)))
    #cv2.imwrite('favorleft.png', favorleft)
    return result


def rightjoin(left, right):
    result = np.array(left)
    np.copyto(result, right, where=right != 0)
    return result


def meanjoin(left, right):
    maskedleft, maskedright = mask(left, right)
    mean = np.ma.mean([maskedleft, maskedright], axis=0)
    #cv2.imwrite('mean.png', mean)
    return mean


def interlacejoin(left, right):
    interlace = np.empty((left.shape[0] * 2, left.shape[1], 3))
    interlace[::2, :, :] = left
    interlace[1::2, :, :] = right
    #print("Interlace join PSNR +inf vs +inf")
    #cv2.imwrite('interlace.png', interlace)
    return interlace


def deltajoin(left, right, quant=1):
    delta = np.empty((left.shape[0] * 2, left.shape[1], 3))
    delta[::2, :, :] = left
    delta[1::2, :, :] = ((np.array(right, dtype=np.int32) - left) / quant) + 127
    #recovered = np.array(left + ((np.array(delta[1::2, :, :], dtype=np.int32) - 127) * quant), dtype=np.uint8)
    #print("Delta join PSNR +inf vs %s" % psnr(maskedright, recovered))
    #cv2.imwrite('delta.png', delta)
    return delta

if __name__ == '__main__':
    leftfilename = sys.argv[1] #"p0/%03d.png"

    rightfilename = sys.argv[2] #"p30/%03d.png"
    outputdirectory = sys.argv[3]
    #count = 91

    leftvideo = cv2.VideoCapture(leftfilename)
    rightvideo = cv2.VideoCapture(rightfilename)

    lresult, left = leftvideo.read()
    rresult, right = rightvideo.read()

    if not lresult or not rresult:
        print 'Fail'
        exit(1)

    #left = cv2.imread(leftfilename % 1)
    left = cv2.copyMakeBorder(left, left.shape[0]/2, left.shape[0]/2, left.shape[1]/2, left.shape[1]/2, cv2.BORDER_CONSTANT, 0)
    H, Hi = homography(left, right) #cv2.imread(rightfilename % 1))
    partitions = partition(H, Hi, left)
    print partitions
    i = 0

    while lresult and rresult:
    #for i in range(1, count + 1):
        print i
        i += 1
        if i == 201: exit(0)
        lresult, left = leftvideo.read()
        rresult, right = rightvideo.read()
        if not lresult or not rresult:
            break

        #left = cv2.imread(leftfilename % i)
        left = cv2.copyMakeBorder(left, left.shape[0]/2, left.shape[0]/2, left.shape[1]/2, left.shape[1]/2, cv2.BORDER_CONSTANT, 0)
        #right = cv2.imread(rightfilename % i)
        cutleft, (overlapleft, overlapright), cutright = cut(left, right, Hi, partitions)
        #output, overlap, cutleft, cutright = cut(left, right, Hi, partitions)
        #cv2.imwrite(os.path.join(outputdirectory, 'overlapleft%03d.png' % i), overlapleft)
        #cv2.imwrite(os.path.join(outputdirectory, 'overlapright%03d.png' % i), overlapright)
        cv2.imwrite(os.path.join(outputdirectory, 'left%03d.png' % i), cutleft)
        cv2.imwrite(os.path.join(outputdirectory, 'right%03d.png' % i), cutright)

        cv2.imwrite(os.path.join(outputdirectory, 'overlapright%03d.png' % i), overlapright)

        result = leftjoin(overlapleft, overlapright)
        cv2.imwrite(os.path.join(outputdirectory, 'leftjoin%03d.png' % i), result)

        result = rightjoin(overlapleft, overlapright)
        cv2.imwrite(os.path.join(outputdirectory, 'rightjoin%03d.png' % i), result)

        result = meanjoin(overlapleft, overlapright)
        cv2.imwrite(os.path.join(outputdirectory, 'meanjoin%03d.png' % i), result)

        result = interlacejoin(overlapleft, overlapright)
        cv2.imwrite(os.path.join(outputdirectory, 'interlacejoin%03d.png' % i), result)

        result = deltajoin(overlapleft, overlapright)
        cv2.imwrite(os.path.join(outputdirectory, 'deltajoin%03d.png' % i), result)

        result = deltajoin(overlapleft, overlapright, quant=4)
        cv2.imwrite(os.path.join(outputdirectory, 'delta4join%03d.png' % i), result)


'''
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
'''
