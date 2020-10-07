from random import randint

HEVC = 0
H264 = 1
FORMATS = 1

goal = H264
toyproblem = [{'id': 0, 'start': 0, 'end': 8, 'format': HEVC},
              {'id': 1, 'start': 1, 'end': 5, 'format': H264},
              {'id': 2, 'start': 5, 'end': 10, 'format': HEVC},
              {'id': 3, 'start': 3, 'end': 7, 'format': HEVC},
              {'id': 4, 'start': 8, 'end': 12, 'format': H264}]


def make_problem(n, end=100):
  goal = randint(0, FORMATS)
  root = {'id': 0, 'start': 0, 'end': end, 'format': randint(0, FORMATS)}
  problem = [root]
  for i in range(1, n):
    start = randint(0, end - 1)
    end = randint(start + 1, end)
    format = randint(0, FORMATS)
    entry = {'id': i, 'start': start, 'end': end, 'format': format}
    problem.append(entry)
  return goal, problem


def weight(start, end, interval, goal_format):
  if interval['format'] == goal_format:
    return 0
  elif start == interval['start']:
    return 10 + (end - start)
  else:
    return 2 + (end -start)


def missing(interval, start, end, solution):
    applicable = [s for s in solution if s[0] == interval['id'] and s[1] < start]
    applicable = sorted(applicable, key=lambda a: a[1])
    applicable = applicable[-1:] if applicable else []
    #print('    a: %d: %s' % (interval['id'], str(applicable)))
    #current = interval['start']
    current = applicable[-1][2] if applicable else interval['start']
    gaps = []
    while current < start:
        #print(current)
        if not applicable:
            if current > end:
                print('error! %d %d' % (current, end))
                exit(1)
            gaps.append((current, end))
            current = end
        elif applicable[0][2] != current:
            if current > applicable[0][2]:
                print(interval)
                print(start, end)
                print(applicable)
                print('error#2 %d %d' % (current, applicable[0][2]))
                exit(1)
            gaps.append((current, applicable[0][1]))
            current = applicable[0][1]
        else:
            current = applicable[0][2]
            applicable.pop()

    weight = 0
    for gap in gaps:
       if gap[0] == interval['start']:
           #print('missing iframe')
           weight += 10 + (gap[1] - gap[0])
       else:
           weight += gap[1] - gap[0]
    #if gaps:
    #  print(gaps)
    #  print(weight)
    #  exit(1)
    return weight


def missingreversedxxx(interval, start, end, solution):
    id = interval['id']
    if id in [s[0] for s in solution]:
        return 0
    else:
        return missing(interval, start, end, [])


def missingreversed(interval, start, end, solution):
    id = interval['id']
    if id in [s[0] for s in solution]:
        return - missing(interval, start, end, [])
    else:
        return missing(interval, start, end, [])


def permutations(intervals, goal_format, start=-1):
  if start == -1:
    start = min([i['start'] for i in intervals])
  end = min([i['start'] for i in intervals if i['start'] > start] +
            [i['end'] for i in intervals if i['end'] > start] +
            [99999999999])

  if end < 999999:
    options = [i for i in intervals if i['start'] <= start and i['end'] >= end]
    solutions = []
    tail = permutations(intervals, goal_format, end)

    for o in options:
      for t in tail:
        solutions.append([(o['id'], start, end)] + t)

    return solutions

  return [[]]


def optimal(intervals, goal_format):
  weighted = []
  solutions = permutations(intervals, goal_format)
  for s in solutions:
    w = 0
    for i, start, end in s:
        interval = intervals[i]
        w += weight(start, end, interval, goal_format) + missing(interval, start, end, s)
    weighted.append([w, s])

  weighted = sorted(weighted, key=lambda w: w[0])
  #for w in weighted:
  #  print(w)

  best = weighted[0][0]
  return [s for s in weighted if s[0] == best]
  #return [] #s #weighted[-1]

def greedy(intervals, goal_format, n):
  opt = [0] * (n+1)
  opt[0] = 0
  solution = []

  start = min([i['start'] for i in intervals])
  end = min([i['start'] for i in intervals if i['start'] > start])

  for j in range(n + 1):
    #print(start, end)

    options = [i for i in intervals if i['start'] <= start and i['end'] >= end]

    opt[j] = min([weight(start, end, o, goal_format) + missing(o, start, end, solution) + opt[j - 1] for o in options] + [9999999])
    candidates = [(o['id'], start, end) for o in options if opt[j] == weight(start, end, o, goal_format) + missing(o, start, end, solution) + opt[j - 1]]
    solution.append(candidates[0])
    #solution.append([(o['id'], start, end) for o in options if opt[j] == weight(start, end, o, goal_format) + missing(o, start, end, solution) + opt[j - 1]][0])

    #print([o['id'] for o in options])
    #print('    w%d' % opt[j])
    #print('    s' + str(solution))

    start = end
    end = min([i['start'] for i in intervals if i['start'] > start] +
              [i['end'] for i in intervals if i['end'] > end] +
              [99999999999])

    if end > 99999999:
        break

  return opt[j], solution

def greedyback(intervals, goal_format, n):
    opt = [0] * (n+2)
    opt[0] = 0
    solution = []

    end = max([i['end'] for i in intervals])
    start = max([i['end'] for i in intervals if i['end'] < end])

    for j in range(n, -1, -1):
        print(j, start, end)

        options = [i for i in intervals if i['start'] <= start and i['end'] >= end]

        opt[j] = min([weight(start, end, o, goal_format) + missingreversed(o, start, end, solution) + opt[j + 1] for o in options] + [9999999])
        candidates = [(o['id'], start, end) for o in options if opt[j] == weight(start, end, o, goal_format) + missingreversed(o, start, end, solution) + opt[j + 1]]

        #if start == 14:
        #    for o in options:
        #        print(o)
        #        print(weight(start, end, o, goal_format), missingreversed(o, start, end, solution), opt[j + 1])
        #    print(options)
        #    exit(1)
        if len(candidates) > 1:
            print('***')
            exit(1)
        solution.insert(0, candidates[0])

        end = start
        start = max([i['start'] for i in intervals if i['start'] < start] +
                    [i['end'] for i in intervals if i['end'] < start] +
                    [-1])

        if start < 0:
            break

    print(opt)

    return opt[j], solution


if __name__ == '__main__':
    #goal = H264
    size = 5
    end = 100
    #goal, problem = make_problem(size, end)

    #goal = 1
    #problem = [{'id': 0, 'start': 0, 'end': 100, 'format': 0}, {'id': 1, 'start': 95, 'end': 101, 'format': 1}, {'id': 2, 'start': 26, 'end': 36, 'format': 2}, {'id': 3, 'start': 33, 'end': 35, 'format': 2}, {'id': 4, 'start': 14, 'end': 22, 'format': 1}]
    problem = [{'start': 0, 'end': 100, 'id': 0, 'format': 0}, {'start': 98, 'end': 100, 'id': 1, 'format': 0}, {'start': 87, 'end': 98, 'id': 2, 'format': 0}, {'start': 40, 'end': 45, 'id': 3, 'format': 1}, {'start': 22, 'end': 39, 'id': 4, 'format': 0}]

    print('goal = %d' % goal)
    print('problem = %s' % problem)


    #gweight, gsol = greedy(problem, goal, 10)
    gweight, gsol = greedyback(problem, goal, 10)
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