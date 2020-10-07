from random import randint
from itertools import groupby
from greedy import optimal, permutations, toyproblem

HEVC = 0
H264 = 1
FORMATS = 1


def weight(start, end, interval, goal_format):
    if interval['format'] == goal_format:
        return 0
    elif start == interval['start']:
        return 10 + (end - start)
    else:
        return 2 + (end - start)


def partitions(intervals, goal_format):
  partitions = permutations(intervals, goal_format)
  flattened = sorted({o for p in partitions for o in p}, key=lambda o: (o[0], o[1]))
      #list(set(reduce(sum, map(list, partitions), [])))
  groups = groupby(flattened, lambda p: p[0])
  partitions = []
  for id, group in groups:
      group = list(group)
      for i in range(len(group)):
        for j in range(i, len(group)):
            partitions.append((id, group[i][1], group[j][2]))
  return partitions


def solve(parts, points, weights, memo, partindex, pointindex):
    if partindex == len(parts):
        if pointindex == len(points):
            return 0
        else:
            return 99999999999999999

    if pointindex == len(points):
        return 0
    if (partindex, pointindex) in memo:
        return memo[partindex, pointindex]

    answer = solve(parts, points, weights, memo, partindex + 1, pointindex)

    if parts[partindex][1] >= points[pointindex] <= parts[partindex][2]:
        weight = weights[partindex]
        end = parts[partindex][2]
        partindex += 1
        while partindex < len(parts) and parts[partindex][1] < end:
            partindex += 1
        pointindex += 1
        while pointindex < len(points) and points[pointindex] < end:
            pointindex += 1
        print weight
        answer = min(answer, weight + solve(parts, points, weights, memo, partindex, pointindex))

    memo[partindex, pointindex] = answer
    return answer

def greedy(intervals, goal_format):
    parts = sorted(partitions(intervals, goal_format), key=lambda p: p[1])
    points = list({p[1] for p in parts}.union({p[2] for p in parts}))
    w = [0] + map(lambda p: weight(p[1], p[2], intervals[p[0]], goal_format), parts)
    n = len(parts) - 1
    memo = {}

    ans = solve(parts, points, w, memo, 0, 0)
    print memo
    return 0, []



def interval(intervals, goal_format):
    parts = sorted(partitions(intervals, goal_format), key=lambda p: p[2])
    w = [0] + map(lambda p: -weight(p[1], p[2], intervals[p[0]], goal_format), parts)
    n = len(parts) - 1

    opt = [0] * (n+1)
    solution = []

    def p(j):
        for i in range(j, 0, -1):
          if parts[i][2] <= parts[j][1]:
              prevs = [0]
              value = parts[i][2]
              while i >= 0 and parts[i][2] == value:
                  prevs.append(i)
                  i -= 1
              return prevs
        return [0]

    opt[0] = 0
    for j in range(1, n + 1):
        #opt[j] = max([w[j] + opt[pj] for pj in p(j)] + [opt[j-1]])
        opt[j] = min([w[j] + opt[pj] for pj in p(j)]) # + [opt[j-1]])

    def recurse(j):
        if j == 0:
            pass #print 'done'
        else:
            for pj in p(j):
              if w[j] + opt[pj] > opt[j-1]:
                print w[j], parts[j]
                recurse(pj)
                break
            else:
              recurse(j-1)

    print opt
    recurse(n)
    return opt[j], solution


if __name__ == '__main__':
    #goal = H264
    size = 5
    end = 100
    #goal, problem = make_problem(size, end)

    goal = 1
    #problem = [{'id': 0, 'start': 0, 'end': 100, 'format': 0}, {'id': 1, 'start': 95, 'end': 101, 'format': 1}, {'id': 2, 'start': 26, 'end': 36, 'format': 2}, {'id': 3, 'start': 33, 'end': 35, 'format': 2}, {'id': 4, 'start': 14, 'end': 22, 'format': 1}]
    #problem = [{'start': 0, 'end': 100, 'id': 0, 'format': 0}, {'start': 98, 'end': 100, 'id': 1, 'format': 0}, {'start': 87, 'end': 98, 'id': 2, 'format': 0}, {'start': 40, 'end': 45, 'id': 3, 'format': 1}, {'start': 22, 'end': 39, 'id': 4, 'format': 0}]
    problem = toyproblem

    print('goal = %d' % goal)
    print('problem = %s' % problem)


    gweight, gsol = greedy(problem, goal)
    #gweight, gsol = interval(problem, goal)
    oweight, osol = optimal(problem, goal)[0]


    #g 30 [(0, 0, 1), (1, 1, 3), (1, 3, 5), (0, 5, 7), (0, 7, 8), (4, 8, 10), (4, 10, 12)]
    #o 26 [(0, 0, 1), (1, 1, 3), (1, 3, 5), (2, 5, 7), (2, 7, 8), (4, 8, 10), (4, 10, 12)]
    print('g %2d %s' % (gweight, gsol))
    print('o %2d %s' % (oweight, osol))

    gtotal, ototal = 0, 0

    for g, o in zip(gsol, osol):
      print '*' if g[0] != o[0] else ' ', problem[g[0]], problem[o[0]]

      gid, gstart, gend = g
      gint = problem[gid]
      gtotal += weight(gstart, gend, gint, goal) +  missing(gint, gstart, gend, gsol)
      print('g weight: %2d+%2d=%2d' % (weight(gstart, gend, gint, goal), missing(gint, gstart, gend, gsol), weight(gstart, gend, gint, goal) +  missing(gint, gstart, gend, gsol)))

      oid, ostart, oend = o
      oint = problem[oid]
      ototal +=  weight(ostart, oend, oint, goal) + missing(oint, ostart, oend, osol)
      print('o weight: %2d+%2d=%2d' % (weight(ostart, oend, oint, goal), missing(oint, ostart, oend, osol), weight(ostart, oend, oint, goal) + missing(oint, ostart, oend, osol)))

    print('Optimal %d vs greedy %d' % (ototal, gtotal))