#!/usr/bin/env python
# -*- coding=utf-8 -*-

class A(object): pass
class B(A): pass
class C(A): pass
class D(A): pass
class E(B, C): pass
class F(C, D): pass
class H(E, F): pass

print A.mro()
print B.mro()
print C.mro()
print D.mro()
print E.mro()
print F.mro()
print(H.mro())
