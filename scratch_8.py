def func5(line):
    list_ = []
    num = 0
    for c in line:
        if c not in ('-', '+'):
            num = num * 10 + int(c)
        else:
            list_.append(num)
            list_.append(c)
            num = 0
    list_.append(num)

    ans, curr = 0, 0
    flag = False
    for c in list_:
        if c == '+':
            pass
        elif c == '-':
            if flag:
                ans -= curr
            else:
                ans += curr
            flag = True
            curr = 0
        else:
            curr += c
    if flag:
        ans -= curr
    else:
        ans += curr
    return ans


import sys

line = sys.stdin.readline().strip()
print(func5(line))
print(func5('55-20+36+100-70+50'))
print(func5('55-50+30'))
print(func5('10+30+40+20'))
print(func5('00009-00008'))
