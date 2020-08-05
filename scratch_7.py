def func4(cmds):
    s, x, y = 'S', 0, 0
    count = 0
    for i in range(4):
        for cmd in cmds:
            for c in cmd:
                x, y, s = process(c, x, y, s)
        if x == 0 and y == 0:
            count += 1
            break
    return "bounded" if count > 0 else "unbounded"


def process(c, x, y, s):
    map_ = {
        'S': {
            'S': (0, 1),
            'X': (0, -1),
            'Z': (-1, 0),
            'Y': (1, 0),
        },
        'R': {
            'S': 'Y',
            'X': 'Z',
            'Z': 'S',
            'Y': 'X'
        },
        'L': {
            'S': 'Z',
            'X': 'Y',
            'Z': 'X',
            'Y': 'S'
        }
    }
    action = map_[c][s]
    if c == 'S':
        x += action[0]
        y += action[1]
    else:
        s = action
    return x, y, s


import sys

line = sys.stdin.readline()
cmds = []
n = int(line)
for _ in range(n):
    line = sys.stdin.readline().strip()
    cmds.append(line)
print(func4(cmds))
