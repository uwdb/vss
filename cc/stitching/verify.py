import sys
import os
import cv2
import numpy as np
from stitchvideo2 import *
from os.path import getsize


def join_split(overlap, shape, H, Hi, (p0, right, p1, (ymin, ymax))):
    #print shape
    #print partitions
    left = overlap[overlap.shape[0]/2 - shape[0]/2:overlap.shape[0]/2 + shape[0]/2, :shape[1], :]

    #print shape
    #print partitions
    ypadding = int((2.0*shape[0] - overlap.shape[0])/2.0)
    xpadding = 2*shape[1] - p0 - overlap.shape[1]
    poverlap = cv2.copyMakeBorder(overlap, ypadding, ypadding, p0, xpadding, cv2.BORDER_CONSTANT, 0)
    #print poverlap.shape
    
    #unwarped = cv2.warpPerspective(poverlap, H, tuple(reversed(poverlap.shape[:2])))
    unwarped = cv2.warpPerspective(poverlap, H, tuple(reversed(shape[:2])))
    right = unwarped[:, :p1, :]
    #print right.shape

    #print unwarped.shape
    #cv2.imwrite('foo3.png', right)
    #cv2.imwrite('foo2.png', poverlap)
    #cv2.imwrite('foo.png', unwarped)
    #exit(1)
    #right = None

    return left, right


def interlace_split(overlap, shape, H, Hi, (p0, right, p1, (ymin, ymax))):
    #print shape
    #print partitions
    #print overlap.shape
    #print shape
    #print (overlap.shape[0]/2), shape[0]*2, (overlap.shape[0]/2) - (shape[0]*2)
    #print (overlap.shape[0]/2 + shape[0]*2)
    left = overlap[(overlap.shape[0]/2 - shape[0]):(overlap.shape[0]/2 + shape[0]):2, :shape[1], :]
    #print left.shape


    roverlap = overlap[1+(overlap.shape[0]/2 - shape[0]):(overlap.shape[0]/2 + shape[0]):2, :shape[1], :]
    ypadding = int((2.0*shape[0] - roverlap.shape[0])/2.0)
    xpadding = 2*shape[1] - p0 - roverlap.shape[1]
    poverlap = cv2.copyMakeBorder(roverlap, ypadding, ypadding, p0, xpadding, cv2.BORDER_CONSTANT, 0)
    #print poverlap.shape

    #unwarped = cv2.warpPerspective(poverlap, H, tuple(reversed(poverlap.shape[:2])))
    unwarped = cv2.warpPerspective(poverlap, H, tuple(reversed(shape[:2])))
    right = unwarped[:, :p1 - 1, :]
    #print right.shape

    #right = None
    return left, right


def verify(name, dir, overlapfilename, (ltruthfilename, rtruthfilename), split):
    tsize = float(getsize(ltruthfilename) + getsize(rtruthfilename))
    lsize = getsize(os.path.join(dir, 'left.h265')) if os.path.exists(os.path.join(dir, 'left.h265')) else 0
    osize = getsize(os.path.join(dir, overlapfilename)) if os.path.exists(os.path.join(dir, overlapfilename)) else 0
    rsize = getsize(os.path.join(dir, 'right.h265')) if os.path.exists(os.path.join(dir, 'right.h265')) else 0

    print '%s: %d%% compression benefit' % (name, 100 * (tsize - (lsize + osize + rsize)) / tsize)

    ltruth = cv2.VideoCapture(ltruthfilename)
    rtruth = cv2.VideoCapture(rtruthfilename)
    left = cv2.VideoCapture(os.path.join(dir, 'left.h265'))
    right = cv2.VideoCapture(os.path.join(dir, 'right.h265'))
    overlap = cv2.VideoCapture(os.path.join(dir, overlapfilename))
    lresult, rresult, oresult, ltresult, rtresult = True, True, True, True, True
    lpsnr, rpsnr = [], []
    H = None

    while lresult and rresult and oresult and ltresult and rtresult:
        lresult, lframe = left.read() if lsize > 0 else (True, np.array([0, 0]))
        rresult, rframe = right.read()
        oresult, oframe = overlap.read()
        ltresult, ltframe = ltruth.read()
        rtresult, rtframe = rtruth.read()

        if H is None:
            pltframe = cv2.copyMakeBorder(ltframe, ltframe.shape[0]/2, ltframe.shape[0]/2, ltframe.shape[1]/2, ltframe.shape[1]/2, cv2.BORDER_CONSTANT, 0)
            H, Hi = homography(pltframe, rtframe)
            partitions = partition(H, Hi, pltframe)

        if lresult and rresult and oresult and ltresult and rtresult:
            loverlap, roverlap = split(oframe, ltframe.shape, H, Hi, partitions)
            #print lframe.shape, loverlap.shape
            lcombined = np.concatenate([lframe, loverlap], axis=1)
            rcombined = np.concatenate([rframe, roverlap], axis=1)
            #print ltframe.shape, lcombined.shape
            #rcombined = np.hstack([rframe, roverlap])
            lpsnr.append(psnr(ltframe, lcombined))
            rpsnr.append(psnr(rtframe, rcombined))
            #print("%s: PSNR %d vs %d" % (name, lpsnr[-1] or -1, 0)) #psnr(rtframe, rtframe) or -1))
            #cv2.imwrite('lfoo.png', lframe)
            #cv2.imwrite('rfoo.png', loverlap)
            #cv2.imwrite('foo.png', lcombined)
            #return

    lpsnr = [r for r in lpsnr if r]
    rpsnr = [r for r in rpsnr if r]
    print("%s: Mean PSNR left %d right %d" % (name, sum(lpsnr) / (len(lpsnr) or 1), sum(rpsnr) / (len(rpsnr) or 1)))
    return sum(lpsnr) / (len(lpsnr) or 1), sum(rpsnr) / (len(rpsnr) or 1)

dir = sys.argv[3]
truthfilenames = sys.argv[1], sys.argv[2] #('p0/out.h265', 'p30/out.h265')

#verify('mean', dir, 'meanjoin.h265', truthfilenames, join_split)
verify('left', dir, 'leftjoin.h265', truthfilenames, join_split)
#verify('right', dir, 'rightjoin.h265', truthfilenames, join_split)
#verify('interlace', dir, 'interlacejoin.h265', truthfilenames, interlace_split)
#verify('delta', dir, 'deltajoin.h265', truthfilenames, mean_split)
#verify('delta4', dir, 'delta4join.h265', truthfilenames, mean_split)
