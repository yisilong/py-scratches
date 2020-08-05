#!/usr/bin/env python
# -*- coding=utf-8 -*-

# Author: likebeta <ixxoo.me@gmail.com>
# Create: 2016-10-05

def calc_pages(total, c_p):
    if total == 0 or c_p < 1 or c_p > total:
        return False, [], False

    # 最多显示10个页码
    left_page = c_p - 1
    right_page = c_p + 1
    pages = [c_p]
    while len(pages) < 10:
        if left_page >= 1:
            pages.insert(0, left_page)
            left_page -= 1
        if len(pages) < 10 and right_page <= total:
            pages.append(right_page)
            right_page += 1
        if left_page < 1 and right_page > total:
            break
    return c_p > 1, pages, c_p < total


for i in range(1, 12):
    print i, calc_pages(100, i)

for i in range(90, 101):
    print i, calc_pages(100, i)