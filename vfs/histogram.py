import logging
import numpy as np
import cv2
from sklearn.cluster import *
from sklearn.preprocessing import maxabs_scale
from vfs.engine import VFS
from vfs.videoio import read_first_frame


class Histogram(object):
    BUCKETS = 2

    @classmethod
    def create(cls, filename, shape, codec):
        frame = read_first_frame(filename)
        #with VideoReader(filename, shape, codec, limit=1) as reader:
        #    frame = reader.read()

        assert(frame is not None)

        return cv2.calcHist([frame], [0, 1, 2], None, [cls.BUCKETS, cls.BUCKETS, cls.BUCKETS],
                            [0, 256, 0, 256, 0, 256]).flatten(), frame

    @classmethod
    def cluster_all(cls):
        data = VFS.instance().database.execute(
            'SELECT id, histogram FROM gops WHERE NOT histogram IS NULL AND examined != 9999999').fetchall()
        if not data or len(data) == 1:
            return 0

        ids, histograms = zip(*data)
        id_array = np.array(ids)
        scaled_histograms = maxabs_scale(np.vstack(histograms))
        clusters = Birch(n_clusters=len(histograms)).fit(scaled_histograms)
        cluster_count = 0

        for cluster_id in range(max(clusters.labels_) + 1):
            rows = id_array[clusters.labels_ == cluster_id]
            if len(rows) > 1:
                cluster_count += 1
                VFS.instance().database.executebatch(
                    'UPDATE gops SET cluster_id = {} where id = {}'.format(int(cluster_id) + 1, int(row_id))
                    for row_id in rows)

        return cluster_count
#                for row_id in rows:
#                    VFS.instance().database.executebatch([
#                        'UPDATE gops SET cluster_id = ? where id = ?', (cluster_id, int(row_id))])

'''
INPUTS = 10
BUCKETS = 2
hists = np.zeros((INPUTS, BUCKETS**3 + 1), dtype=np.float32)

for index in range(INPUTS):
    filename = "inputs/p{}.mp4".format(index + 1)
    video = cv2.VideoCapture(filename)
    ret, frame = video.read()

    hist = cv2.calcHist([frame], [0, 1, 2], None, [BUCKETS,BUCKETS,BUCKETS], [0, 256, 0, 256, 0, 256])
    hists[index] = np.hstack([[index], hist.flatten()])
    cv2.imwrite('out/hist%d.png' % index, frame)
    gop = Gop.add(filename, 0)
    Histogram.add(gop.id)

Histogram.get_best()
Histogram.get_best()

#hists[index] = hists[0]
c = 15
hists = maxabs_scale(hists)
for i in range(c):
    hists = np.vstack([hists, hists])
for i in range(hists.shape[0]):
    hists[i, 0] = i

#criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#flags = cv2.KMEANS_RANDOM_CENTERS
#ret, labels, centers = cv2.kmeans(hists, 5, None, criteria, 10, flags)
#print(labels)

#next = closely.solve(hists, n=10)
#print(next)

#0,1,2
#3
#4,6,7,8
#clustering = Birch(branching_factor=50, n_clusters=MeanShift(cluster_all=False)).fit(hists)
clustering = Birch(n_clusters=INPUTS).fit(hists[:, 1:])
#print(clustering)

group0 = hists[clustering.labels_ == 0]
print(group0)
#next = closely.solve(hists, n=2)
#print(next)
'''