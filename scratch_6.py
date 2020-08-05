def isPowerOfTwo(n):
    if n == 1:
        return 1
    if n == 0 or n % 2 == 1:
        return 0
    return isPowerOfTwo(n // 2)


def func2(n):
    if n < 3:
        return 1
    pre1 = 1
    pre2 = 1
    i = 3
    result = 0
    while i <= n:
        result = pre1 + pre2
        pre2 = pre1
        pre1 = result
        i += 1
    return result


def func3(nums, src='5'):
    call = {
        '1': (0, 0),
        '2': (0, 1),
        '3': (0, 2),
        '4': (1, 0),
        '5': (1, 1),
        '6': (1, 2),
        '7': (2, 0),
        '8': (2, 1),
        '9': (2, 2),
        '*': (3, 0),
        '0': (3, 1),
        '#': (3, 2),
    }

    dis = 0
    for dst in nums:
        dis += (abs(call[src][0] - call[dst][0]) + abs(call[src][1] - call[dst][1]))
        src = dst
    return dis


import sys

line = sys.stdin.readline().strip()
# n = int(line)
print(func3(line, '5'))