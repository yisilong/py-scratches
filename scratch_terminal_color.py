#!/usr/bin/env python3

styles_group = {
    "1": {
        # styles
        "bold": [1, 22],
        "italic": [3, 23],
        "underline": [4, 24],
        "inverse": [7, 27],
        # grayscale
        "white": [37, 39],
        "grey": [90, 39],
        "black": [90, 39],
        # colors
        "blue": [34, 39],
        "cyan": [36, 39],
        "green": [32, 39],
        "magenta": [35, 39],
        "red": [91, 39],
        "yellow": [33, 39],
    },
    "2": {
        "30": [30, 0],
        "31": [31, 0],
        "32": [32, 0],
        "33": [33, 0],
        "34": [34, 0],
        "35": [35, 0],
        "36": [36, 0],
        "37": [37, 0],
        "38": [38, 0],
        "39": [39, 0],
    }
}


def colorize(s, style):
    return f'\x1B[{style[0]}m{s}\x1B[{style[1]}m'


for k, styles in styles_group.items():
    print(f'--------------{k}-------------')
    for name, style in styles.items():
        print(f'{name}: {colorize("hello world", style)}')
