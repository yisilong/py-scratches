#!/usr/bin/env python
# -*- coding=utf-8 -*-

class BaseA(object):
    def __init__(self):
        print("enter BaseA")
        super(BaseA, self).__init__()
        print("leave BaseA")


class BaseB(object):
    def __init__(self):
        print("enter BaseB")
        super(BaseB, self).__init__()
        print("leave BaseB")


class A(BaseA):
    def __init__(self):
        print("enter A")
        super(A, self).__init__()
        print("leave A")


class B(BaseB):
    def __init__(self):
        print("enter B")
        super(B, self).__init__()
        print("leave B")


class A0(BaseA):
    def __init__(self):
        print("enter A0")
        super(A0, self).__init__()
        print("leave A0")


class B0(BaseB):
    def __init__(self):
        print("enter B0")
        super(B0, self).__init__()
        print("leave B0")


class C(B, A, B0, A0):
    def __init__(self):
        print("enter C")
        super(B0, self).__init__()
        print("leave C")


print(C.mro())
c = C()
