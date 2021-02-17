from itertools import islice
from collections import defaultdict
from itertools import groupby
import logging
import numpy as np
import cv2

from vfs.engine import VFS
from vfs.utilities import log_runtime


class Descriptor(object):
    FLANN_INDEX_LSH = 6
    FLANN_INDEX_KDTREE = 0
    DEFAULT_MATCHES_REQUIRED = 10 #20
    LOWE_THRESHOLD = 0.7

    _matchers = {}
    _feature_fast = cv2.ORB_create()
    _feature_slow = cv2.ORB_create() #1000)

    @classmethod
    def create(cls, frame, fast=False):
        #logging.info('Adding descriptor')

        #orb = cv2.ORB_create()
        if not fast:
            keypoints, descriptors = cls._feature_fast.detectAndCompute(frame, None)
        else:
            keypoints, descriptors = cls._feature_slow.detectAndCompute(frame, None)

        return np.float32([keypoint.pt for keypoint in keypoints]).reshape(-1,1,2), descriptors #.astype(np.float32)

    @classmethod
    def closest_match(cls, epoch, gop, matches_required=DEFAULT_MATCHES_REQUIRED, radius=400): #400):
        from vfs.gop import Gop

        with log_runtime("Joint compression candidate selection"):
            cluster_gops = VFS.instance().database.execute("SELECT id, filename, descriptors FROM gops WHERE cluster_id = ? AND physical_id != ? AND descriptors IS NOT NULL AND joint = 0 AND examined <= ? AND NOT EXISTS (SELECT id FROM gop_joint_aborted WHERE gop1 = ? AND gop2 = id)", (gop.cluster_id, gop.physical_id, epoch, gop.id)).fetchall()
            candidate_gops = []
            for gop2_id, gop2_filename, gop2_descriptors in cluster_gops:
                success, matches = cls.adhoc_match(gop.descriptors, gop2_descriptors)
                if success and len(matches) > matches_required:
                    candidate_gops.append((gop.filename.split('-')[-1] == gop2_filename.split('-')[-1], len(matches), gop2_id, matches))
                    if candidate_gops[-1][1] > 400 or candidate_gops[-1][0]: # Break on "good enough" match to try out
                        break
            candidate_gop = sorted(candidate_gops, reverse=True)[0] if candidate_gops else None
            return [(candidate_gop[2], candidate_gop[3])]

        matcher = cls._get_matcher(epoch, gop)
        physical_map = cls._get_physical_map(gop.cluster_id).get(gop.physical_id, None)
        index_map = cls._get_index_map(gop.cluster_id)
        all_matches = matcher.radiusMatch(queryDescriptors=gop.descriptors, maxDistance=radius)
        good_matches = defaultdict(lambda: [])
        first_matches = {}
        complete = set()

        # For each frame/descriptor pair, find matches that pass the Lowe threshold test
        #all_matches = all_matches[:5000]
        filtered_matches = (m for d in all_matches
                            for m in d
                            if index_map[m.imgIdx] > gop.id and
                               m.imgIdx not in physical_map and
                               not (m.imgIdx, m.queryIdx) in complete)
        #for descriptor_matches in all_matches:
        #    for match in descriptor_matches:
        with log_runtime("Lowes test"):
            for match in filtered_matches:
                #if match.imgIdx not in physical_map and \
                #    index_map[match.imgIdx] > gop.id and \
                #if not (match.imgIdx, match.queryIdx) in complete:
                # First match
                if (match.imgIdx, match.queryIdx) not in first_matches:
                    first_matches[match.imgIdx, match.queryIdx] = match
                # Second match
                else:
                    if first_matches[match.imgIdx, match.queryIdx].distance < cls.LOWE_THRESHOLD * match.distance:
                        good_matches[match.imgIdx].append(first_matches[match.imgIdx, match.queryIdx])
                    del first_matches[match.imgIdx, match.queryIdx]
                    complete.add((match.imgIdx, match.queryIdx))

        # Some matches may not have a second match to apply Lowe's threshold on.
        # Check to see if we should have seen it and count it if so.
        for first_match in first_matches.values():
            if first_match.distance / cls.LOWE_THRESHOLD < radius:
                good_matches[first_match.imgIdx].append(first_match)

        ignore_ids = set(VFS.instance().database.execute('SELECT gop2 FROM gop_joint_aborted WHERE gop1 = ?', gop.id).fetchall())
        best_indexes = [index for index, matches in good_matches.items() if len(matches) >= matches_required and index_map[index] not in ignore_ids]
        #best_ids = [index_map[index] for index in best_indexes if index_map[index] not in ignore_ids]
        #best_id = VFS.instance().database.execute(
        #    'SELECT MIN(id) FROM gops WHERE joint=0 AND id in ({})'.format(','.join(map(str, best_ids)))).fetchone()[0]
        #best_index = max((index for index, matches in good_matches.items() if len(matches) >= matches_required),
        #                 default=None)
        #best = sorted([(index_map[index], good_matches[index]) for index in best_indexes], key=lambda pair: len(pair[1]), reverse=True)
        gops = Gop.get_all(index_map[index] for index in best_indexes)
        best = [(gop, good_matches[index]) for gop, index in zip(gops, best_indexes)]
        best = sorted(best, key=lambda pair: (pair[0].filename.split('-')[-1] == gop.filename.split('-')[-1], len(pair[1])), reverse=True)
        best = best[:len(best)//20 + 1] # Keep top 5%
        best = [(mgop.id, matches) for mgop, matches in best]
        #best = sorted([(mgop.id, cv2.compareHist(gop.histogram, mgop.histogram, cv2.HISTCMP_CHISQR), gop.filename, mgop.filename, matches) for (mgop, matches) in best], key=lambda pair: (len(pair[2]), cv2.compareHist(gop.histogram, pair[0].histogram, cv2.HISTCMP_CHISQR), -pair[0].id), reverse=True)
        #best = sorted([(id, cv2.compareHist(gop.histogram, Gop.get(id).histogram, cv2.HISTCMP_CHISQR), gop.filename, Gop.get(id).filename, matches) for (id, matches) in best], key=lambda pair: (len(pair[2]), cv2.compareHist(gop.histogram, Gop.get(pair[0]).histogram, cv2.HISTCMP_CHISQR), -pair[0]), reverse=True)

        return best

        if best_id is not None:
            return Gop.get(best_id), good_matches[best_indexes[best_ids.index(best_id)]] #best_index]
        #if best_index is not None:
        #    return Gop.get(index_map[best_index]), good_matches[best_index]
            #return Gop.get(VFS.instance().database.execute(
            #    '''SELECT id FROM gops
            #              WHERE cluster_id = ? AND joint = 0
            #              LIMIT 1
            #              OFFSET ?''', (gop.cluster_id, best_index)).fetchone()[0]), good_matches[best_index]
        else:
            return None, None

    @classmethod
    def retrain_matcher(cls, cluster_id):
        del cls._matchers[cluster_id]

    _matcher_fast = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    _matcher_slow = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    @classmethod
    def adhoc_match(cls, descriptors1, descriptors2, fast=True):
        if fast:
            #search_parameters = dict(checks=2)
            #index_parameters = dict(algorithm=cls.FLANN_INDEX_KDTREE, trees=1)
            #matcher = cv2.FlannBasedMatcher(index_parameters, search_parameters)
            #search_parameters = {} #dict(checks=50)
            #index_parameters = dict(algorithm=cls.FLANN_INDEX_LSH)
            #matcher = cv2.FlannBasedMatcher(index_parameters, search_parameters)
            matcher = cls._matcher_fast
            #matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            #search_parameters = dict(checks=50)
            #index_parameters = dict(algorithm=cls.FLANN_INDEX_KDTREE, trees=5)
            #matcher = cv2.FlannBasedMatcher(index_parameters, search_parameters)
            #matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            #matches = matcher.match(descriptors1, descriptors2)
            #return len(matches) > cls.DEFAULT_MATCHES_REQUIRED, matches
            matcher = cls._matcher_slow
            #matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        matches = matcher.knnMatch(queryDescriptors=descriptors1.astype(np.uint8), trainDescriptors=descriptors2.astype(np.uint8), k=2)

        # ratio test as per Lowe's paper
        matches = (m for m in matches if len(m) == 2)
        matches = [m for m, n in matches if m.distance < cls.LOWE_THRESHOLD * n.distance]

        return len(matches) > cls.DEFAULT_MATCHES_REQUIRED, matches

    @classmethod
    def _get_physical_map(cls, cluster_id):
        return cls._matchers[cluster_id][1]

    @classmethod
    def _get_index_map(cls, cluster_id):
        return cls._matchers[cluster_id][2]

    @classmethod
    def _get_matcher(cls, epoch, gop):
        #TODO
        if True or gop.cluster_id not in cls._matchers:
            search_parameters = dict(checks=2)
            index_parameters = dict(algorithm=cls.FLANN_INDEX_KDTREE, trees=1)
            cls._matchers[gop.cluster_id] = cv2.FlannBasedMatcher(index_parameters, search_parameters), {}

        return cls._update_matcher(epoch, gop.id, gop.cluster_id, gop.video().codec, gop.fps)

        '''
            cluster = VFS.instance().database.execute(
                               'SELECT row_number() OVER(ORDER BY id), physical_id, descriptors FROM gops
                                        WHERE cluster_id = ?
                                        ORDER BY physical_id, id', gop.cluster_id).fetchall()
            physical_map = {pid: [rowid for rowid, pid, descriptor in rows]
                            for pid, rows in groupby(cluster, lambda c: c[0])}
            descriptors = [c[2].astype(np.float32) for c in cluster]

            if descriptors:
                matcher.add(descriptors)
                matcher.train()

            cls._matchers[gop.cluster_id] = matcher, physical_map

        return cls._matchers[gop.cluster_id][0]
        '''

    @classmethod
    def _update_matcher(cls, epoch, gop_id, cluster_id, codec, fps):
        matcher = cls._matchers[cluster_id][0]
        old_descriptors = matcher.getTrainDescriptors()
        old_physical_map = cls._get_physical_map(cluster_id)
        offset = len(old_descriptors)

        cluster = VFS.instance().database.execute(
                           'SELECT row_number() OVER(ORDER BY gops.id) + ? - 1, physical_id, gops.id, descriptors '
                           'FROM gops '
                           'INNER JOIN physical_videos '
                           '  ON gops.physical_id = physical_videos.id '
                           'WHERE cluster_id = ? AND (examined <= ? or gops.id = ?) AND codec = ? AND fps = ?'
                           'ORDER BY physical_id, gops.id', (offset, cluster_id, epoch, gop_id, codec, fps)).fetchall()
        new_physical_map = {pid: [rowid for rowid, pid, gid, descriptor in rows]
                        for pid, rows in groupby(cluster, lambda c: c[1])}

        index_map = {rowid: gid for rowid, pid, gid, descriptor in cluster}
        physical_map = {key: old_physical_map.get(key, []) + new_physical_map.get(key, [])
                        for key in set(old_physical_map) | set(new_physical_map)}
        new_descriptors = [c[3].astype(np.float32) for c in cluster]

        if new_descriptors:
            matcher.add(new_descriptors)
            matcher.train()

        cls._matchers[cluster_id] = matcher, physical_map, index_map
        return cls._matchers[cluster_id][0]

'''
FLANN_INDEX_LSH = 6
FLANN_INDEX_KDTREE = 0
THRESHOLD = 0.7 # Lowe et al.
INPUTS = 10 #
DUPLICATES = 100000
FEATURES = 25
MINIMUM_MATCHES_REQUIRED = FEATURES//2

for index in range(INPUTS):
    filename = "inputs/p{}.mp4".format(index + 1)
    Gop.add(filename, 0)

Histogram.cluster_all()
#matcher = Descriptor.train_matcher(0)
#h = Histogram.get_best()
#d = h.gop().descriptor() #
d = Gop.get(1).descriptor
m = d.closest_match()

descriptors = [None] * INPUTS

print('Detecting')
for index in range(INPUTS):
    filename = "inputs/p{}.mp4".format(index + 1)
    video = cv2.VideoCapture(filename)
    ret, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    #orb.setMaxFeatures(FEATURES)
    #sift = cv2.SIFT_create()
    #kaze = cv2.AKAZE_create()
    #kaze.setDescriptorSize(10)
    #kaze.setThreshold(0.0001)
    keypoints, cdescriptors = orb.detectAndCompute(frame, None)
    cdescriptors = cdescriptors.astype(np.float32)
    cdescriptors = cdescriptors[-FEATURES:,:]
    descriptors[index] = cdescriptors

    #add_descriptors(filename)

print('/Detecting')

#index_parameters = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_parameters = dict(checks=2)
index_parameters = dict(algorithm = FLANN_INDEX_KDTREE, trees=1)
#search_parameters = dict(checks=50)   # or pass empty dictionary
matcher = cv2.FlannBasedMatcher(index_parameters, search_parameters)

#matcher = cv2.DescriptorMatcher.create("FlannBased")
matcher.add(descriptors * DUPLICATES)
print('Training')
matcher.train()
print('/Training')

mask = np.ones_like(descriptors)
mask[0] = 0

rmatches = matcher.radiusMatch(queryDescriptors=descriptors[0], maxDistance=250) #, mask=mask)
ims = {index: 0 for index in range(len(descriptors) * DUPLICATES + 1)}
for m in rmatches:
    for n in m:
        ims[n.imgIdx] += 1
hits = {index: h for index, h in ims.items() if h > MINIMUM_MATCHES_REQUIRED}
print('Ims: %s' % {index:h for index, h in list(ims.items())[:10]})
#print('Hits: %s' % hits)
print('Sort: %s' % sorted([i for i,h in hits.items()]))
print('')
#hits = [m for m in matches if m]
#print(len(matches))
#print(len(hits))

def rfoo():
    def bar():
        rmatches = matcher.radiusMatch(queryDescriptors=descriptors[0], maxDistance=250)  # , mask=mask)
    return bar
t = timeit.Timer(rfoo())
print(t.timeit(10))


matches = matcher.knnMatch(queryDescriptors=descriptors[0], k=2) #, mask=mask)
# ratio test as per Lowe's paper
matches = [m for m in matches if len(m) == 2]
matches = [(m, n) for m, n in matches if m.distance < THRESHOLD * n.distance]

if len(matches) < MINIMUM_MATCHES_REQUIRED:
    print('Insufficient matches, should encode separately')


ims = {index: 0 for index in range(INPUTS)}
for m in matches:
    for n in m:
        ims[n.imgIdx + 1] += 1
hits = [index for index, h in ims.items() if h > MINIMUM_MATCHES_REQUIRED]

print('Hits: %s' % hits)

def foo():
    def bar():
        matcher.knnMatch(queryDescriptors=descriptors[0], k=2)
    return bar

t = timeit.Timer(foo())
print(t.timeit(10))

print('Done')
'''