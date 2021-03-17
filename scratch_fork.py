#!/usr/bin/env python3

import os
import sys
import time
import signal
import subprocess


def on_term_handler(signal_num, frame):
    print(f'on signal signal:{signal_num}, pid:{os.getpid()}, ppid:{os.getppid()}')
    sys.exit()


def child_run():
    while True:
        print(f'in child pid:{os.getpid()}')
        time.sleep(3)


def parent_run():
    while True:
        print(f'in parent pid:{os.getpid()}')
        time.sleep(3)


def main():
    signal.signal(signal.SIGTERM, on_term_handler)
    signal.signal(signal.SIGINT, on_term_handler)
    if len(sys.argv) == 1:
        signal.signal(signal.SIGCHLD, signal.SIG_IGN)
        subprocess.Popen([sys.executable, sys.argv[0], 'env=dev', 'type=gate-1'], close_fds=True)
        parent_run()
    else:
        child_run()


if __name__ == "__main__":
    main()
