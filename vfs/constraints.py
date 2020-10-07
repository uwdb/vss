import sys
import json
from typing import List
from collections import defaultdict
import z3
import itertools
from vfs.videoio import encoded
# import random


KEYFRAME_COST = 5
NON_KEYFRAME_COST = 1
GOP_SIZE = 30

class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def intersects(self, other):
        return other.start <= self.start <= other.end \
                or self.start <= other.start <= self.end

    def length(self):
        return self.end - self.start + 1 # +1 because both ends inclusive.


class GOP(object):
    def __init__(self, interval: Interval, target):
        self.interval = interval
        self.target = target


class Fragment(object):
    def __init__(self, interval: Interval, target, _id):
        self.interval = interval
        self.target = target
        self.id = _id

class DecodeDependencies:
    def __init__(self):
        self.num_keyframes = 0
        self.num_p_frames = 0
        self.intersecting_gops = []


class Video(object):
    def __init__(self, gops: List[GOP], fragments: List[Fragment], _id):
        self.gops = gops
        self.fragments = fragments
        self.id = _id
        self.fragment_decode_dependencies: List[DecodeDependencies] = []
        self.gop_to_partial_fragments = defaultdict(list)
        self.find_decode_dependencies()

    def find_decode_dependencies(self):
        gop_index = 0
        for i, fragment in enumerate(self.fragments):
            decode_info = DecodeDependencies()
            # Move the gop_index forward until it starts at least past this fragment start.
            while self.gops[gop_index].interval.end < fragment.interval.start:
                gop_index += 1

            while gop_index < len(self.gops) and self.gops[gop_index].interval.start <= fragment.interval.end:
                # See if the gop index is completely contained in the interval.
                if self.gops[gop_index].interval.start >= fragment.interval.start \
                        and self.gops[gop_index].interval.end <= fragment.interval.end:
                    decode_info.num_keyframes += 1
                    decode_info.num_p_frames += self.gops[gop_index].interval.length() - 1
                else:
                    decode_info.intersecting_gops.append(gop_index)
                    self.gop_to_partial_fragments[gop_index].append(i)

                gop_index += 1
            gop_index -= 1  # Move it back in case the same GOP overlaps with another fragment.
            self.fragment_decode_dependencies.append(decode_info)


def split_up_fragment(start, stop, transition_points):
    sub_fragments = []
    tp_idx = 0
    while transition_points[tp_idx] <= start:
        tp_idx += 1

    begin = start
    while tp_idx < len(transition_points) and transition_points[tp_idx] <= stop:
        end = transition_points[tp_idx]
        sub_fragments.append((begin, end))
        begin = end
        tp_idx += 1

    if begin < stop:
        sub_fragments.append((begin, stop))

    return sub_fragments


def to_frames(s):
    fps = 30
    return round(s * fps)


def build_from_video_info(video_videos, video_fragments, goal_fragment):
    # Build up videos, associated fragments, and goal fragments.
    transition_points = set()
    for fragment in video_fragments:
        transition_points.add(fragment['start'])
        transition_points.add(fragment['end'])

    transition_points.add(goal_fragment[0])
    transition_points.add(goal_fragment[1])
    transition_points = list(transition_points)
    transition_points.sort()

    # Split up each fragment into smaller pieces based on transition points.
    video_to_fragment_info = defaultdict(list)
    for fragment in video_fragments:
        # Find the first transition point that is >= fragment's start.
        start = fragment['start']
        stop = fragment['end']
        source = fragment['source']
        # video_to_fragment_info[source].append(fragment)
        split_up = split_up_fragment(start, stop, transition_points)
        video_to_fragment_info[source] += [{'start': sub[0], 'end': sub[1], 'id': fragment['id']} for sub in split_up]

    goal_subfragments = split_up_fragment(goal_fragment[0], goal_fragment[1], transition_points)

    # print(video_to_fragment_info)

    # Now build Video objects based on GOP size and associated fragments.
    videos = []
    for video in video_videos:
        start = video['start']
        stop = video['end']
        vid_format = video['format']
        video_id = video['id']

        start_frame = to_frames(start)
        stop_frame = to_frames(stop) - 1

        # Build GOPs.
        if stop_frame - start_frame < GOP_SIZE:
            gops = [GOP(Interval(start_frame, stop_frame), vid_format)]
        else:
            gops = [GOP(Interval(start, min(start + GOP_SIZE - 1, stop_frame)), vid_format)
                    for start in range(start_frame, stop_frame, GOP_SIZE)]
        fragments = [Fragment(Interval(to_frames(sub_fragment['start']), to_frames(sub_fragment['end'])-1), vid_format, sub_fragment['id'])
                     for sub_fragment in video_to_fragment_info[video_id]]

        videos.append(Video(gops, fragments, video_id))

    goal_ints = [Interval(to_frames(sub[0]), to_frames(sub[1]) - 1) for sub in goal_subfragments]

    return videos, goal_ints


def is_raw(target):
    return encoded[target]
    #return target > 2


def find_best_intervals(videos: List[Video], goal_intervals: List[Interval], target):
    # For each GOP, find the intervals that are from the same video and cross that GOP.
    # For each goal fragment, create a boolean variable.
    opt = z3.Optimize()

    # Each interval has to be covered.
    cover_vars = []
    for i, interval in enumerate(goal_intervals):
        is_covered = z3.Bool(f'interval-{i}')
        opt.add(is_covered)
        cover_vars.append(is_covered)

    # Each video fragment can be used or not.
    goal_interval_to_video_fragments = defaultdict(list)
    video_fragment_indicators: List[List] = []
    for i, video in enumerate(videos):
        fragment_indicators = []
        for j, fragment in enumerate(video.fragments):
            # Requires that the fragment intervals exactly match some goal interval.
            goal_interval_to_video_fragments[(fragment.interval.start, fragment.interval.end)].append((i, j))

            # Could probably be a bool, but then have to use ite rather than indicator variable.
            fragment_is_used = z3.Int(f'fragment-{i}-{j}')
            opt.add(fragment_is_used >= 0, fragment_is_used <= 1)
            fragment_indicators.append(fragment_is_used)
        video_fragment_indicators.append(fragment_indicators)

    # Each interval is covered iff one fragment covers it. Only one fragment should be picked.
    for i, interval in enumerate(goal_intervals):
        # Find all possible fragments for this interval.
        possible_fragments = goal_interval_to_video_fragments[(interval.start, interval.end)]
        if len(possible_fragments) == 1:
            video_index = possible_fragments[0][0]
            fragment_index = possible_fragments[0][1]
            opt.add(cover_vars[i] == video_fragment_indicators[video_index][fragment_index] > 0)
        elif len(possible_fragments):
            choice1 = possible_fragments[0]
            choice2 = possible_fragments[1]
            pick_one = z3.Xor(video_fragment_indicators[choice1[0]][choice1[1]] > 0, video_fragment_indicators[choice2[0]][choice2[1]] > 0)
            for j in range(2, len(possible_fragments)):
                choice = possible_fragments[j]
                pick_one = z3.Xor(pick_one, video_fragment_indicators[choice[0]][choice[1]] > 0)
            opt.add(cover_vars[i] == pick_one)
        else:
            # Shouldn't really happen, but useful for checking logic.
            assert False
            opt.add(cover_vars[i] == z3.BoolVal(False))

    # Add encode cost for each fragment that is used.
    # If the fragment's target == goal target, then encode cost is 0.
    # Else it's the size of the fragment.
    # If the target format is raw, then switching formats will still be necessary.
    video_fragment_encode_costs: List[List] = []
    # if not is_raw(target):
    for v, video in enumerate(videos):
        fragment_encode_costs = []
        for f, fragment in enumerate(video.fragments):
            encode_cost = z3.Int(f'encode-cost-{v}-{f}')
            opt.add(encode_cost >= 0)
            if fragment.target == target:
                encode_cost = z3.IntVal(0)
            else:
                # There is no lookback cost associated with encoding because it assumes the starting point is raw.
                # should_add_penalty_for_encoding_raw_frame = fragment.interval.length() == 1 and is_raw(fragment.target)
                encode_cost = video_fragment_indicators[v][f] * fragment.interval.length()
            fragment_encode_costs.append(encode_cost)
        video_fragment_encode_costs.append(fragment_encode_costs)


    # Look at each fragment. Decode cost = decode cost for all GOPs that are completely covered + z3 cost for GOPs that are partially covered.
    # If the GOP is in the same format as the target, then no decode is necessary.
    video_decode_and_lookback_costs: List[List] = []
    for v, video in enumerate(videos):
        # Have non-negotiable fragment decode costs.
        fragment_decode_costs = []
        for f, fragment in enumerate(video.fragments):
            fragment_decode_cost = z3.Int(f'fragment-decode-cost-{v}-{f}')
            if is_raw(fragment.target) or fragment.target == target:
                fragment_decode_cost = z3.IntVal(0)
            else:
                fragment_decode_cost = video_fragment_indicators[v][f] *\
                                       (video.fragment_decode_dependencies[f].num_p_frames * NON_KEYFRAME_COST + video.fragment_decode_dependencies[f].num_keyframes * KEYFRAME_COST)
            fragment_decode_costs.append(fragment_decode_cost)

        video_decode_and_lookback_costs.append(fragment_decode_costs)

        gop_decode_costs = []
        for gop_idx, fragment_idxs in video.gop_to_partial_fragments.items():
            # Add the decode cost of the GOP.
            if is_raw(video.gops[gop_idx].target):
                # If the GOP is raw, then there is no decode or lookback cost.
                continue
            elif video.gops[gop_idx].target != target:
                # If the GOPs target doesn't equal the goal target, then we will have to decode as much as necessary for
                # any fragments that are picked and partially lie in this GOP.
                # Start by looking at the furthest fragments, then work towards the ones at the start of the GOP.
                # Reverse the list of fragment indexes so that the latest one comes first.
                # This assumes that the fragments are stored in ascending order.
                fragment_idxs.sort(reverse=True)
                not_later_fragments = z3.And()
                fragment_subcosts = []
                for f_idx in fragment_idxs:
                    associated_fragment = video.fragments[f_idx]
                    associated_gop = video.gops[gop_idx]
                    num_p_frames_to_decode = min(associated_fragment.interval.end, associated_gop.interval.end) - associated_gop.interval.start
                    # TODO: Update to use actual cost.
                    cost = num_p_frames_to_decode * NON_KEYFRAME_COST + 1 * KEYFRAME_COST

                    fragment_cost = z3.If(z3.And(not_later_fragments, video_fragment_indicators[v][f_idx] > 0), cost, 0)

                    fragment_subcosts.append(fragment_cost)
                    not_later_fragments = z3.And(not_later_fragments, video_fragment_indicators[v][f_idx] < 1)
                decode_cost = z3.Sum(fragment_subcosts)
                gop_decode_costs.append(decode_cost)
            else:
                # The gop's target is the same as the goal. We only have to count the cost of decoding up to the start of the last fragment.
                fragment_idxs.sort(reverse=True)
                not_later_fragments = z3.And()
                lookback_costs = []
                for f_idx in fragment_idxs:
                    associated_fragment = video.fragments[f_idx]
                    associated_gop = video.gops[gop_idx]
                    # There will only be a lookback cost if the fragment starts within the GOP.
                    # If a fragment ends partway through a GOP, there is no cost because that GOP can be truncated without
                    # decoding since it's already in the desired format.
                    # Don't have to check ends because we know the fragment intersects the GOP.
                    if associated_fragment.interval.start > associated_gop.interval.start:
                        num_leading_p_frames = associated_fragment.interval.start - associated_gop.interval.start
                        # TODO: Update to use actual cost.
                        cost = num_leading_p_frames * NON_KEYFRAME_COST + 1 * KEYFRAME_COST

                        # I think this should actually be the cost of all of the frames not in fragments but before and
                        # between included fragments.
                        fragment_cost = z3.If(z3.And(not_later_fragments, video_fragment_indicators[v][f_idx] > 0), cost, 0)

                        lookback_costs.append(fragment_cost)

                    not_later_fragments = z3.And(not_later_fragments, video_fragment_indicators[v][f_idx] < 1)
                lookback_cost = z3.Sum(lookback_costs)
                gop_decode_costs.append(lookback_cost)
        video_decode_and_lookback_costs.append(gop_decode_costs)

    flat = list(itertools.chain.from_iterable(video_fragment_encode_costs + video_decode_and_lookback_costs))
    opt.minimize(z3.Sum(flat))

    result = opt.check()
    if result == z3.sat:
        model = opt.model()

        # Get the fragments from each video that should be read.
        video_to_fragments = defaultdict(list)
        fragment_ids = []
        total_num_fragments = 0
        for v, video in enumerate(videos):
            for f, fragment in enumerate(video.fragments):
                if model.eval(video_fragment_indicators[v][f]) == 1:
                    video_to_fragments[video.id].append(fragment.id)
                    fragment_ids.append(fragment.id)
                    total_num_fragments += 1

        # print(video_to_fragments)
        return fragment_ids
    else:
        print(opt.unsat_core())

    return {}


if __name__ == "__main__":
    print(f'GOP size: {GOP_SIZE}')

    with open(sys.argv[1], 'r') as file:
        data = file.read().replace("'", '"')
        json_video = json.loads(data)

    # print(json_video['fragments'])

    video_objects, goal_ints = build_from_video_info(json_video['videos'], json_video['fragments'], (0, 3600))


    fragment_ids = find_best_intervals(video_objects, goal_ints, json_video['goal'])
    print(sorted(fragment_ids))

# json_fragments = json_video['fragments']
# for fragment_id in fragment_ids:
#     print(json_fragments[fragment_id])




