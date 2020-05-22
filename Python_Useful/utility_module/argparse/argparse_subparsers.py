#!/usr/bin/env python
import argparse
import sys

parser = argparse.ArgumentParser(description="This is a demo of the subargparses")
subparsers = parser.add_subparsers()   # add subparsers

# Sub-parser
# Sub-parser的目的是将传入参数用作不同模块。例如该py文件中好2个函数实现2个不同功能。parse将传入参数打包，并基于这些参数调用函数。
# 用sub-parser可以分别打包，分别调用函数。
sbp1 = subparsers.add_parser("sub 1", description="sub 1")
sbp1.add_argument("--aaa", "-a", default=False, action="store_true",
                    help="This is the description of the paramter. If given, the value of the relative variable is True")

sbp2 = subparsers.add_parser("sub 1", description="sub 1")
sbp2.add_argument("--bbb", "-b", type=int, nargs='+',
                    help="This is the description of the paramter. narg='+', '*','?' or N")

args_parsed1, args_unparsed1=sbp1.parse_known_args(sys.argv[1:])   # 提取已定义和未定义的参数，分别保留在第一个和第二个变量中。已定义的提取为Namespace。
args_parsed2, args_unparsed2=sbp2.parse_known_args(sys.argv[1:])   # 提取已定义和未定义的参数，分别保留在第一个和第二个变量中。已定义的提取为Namespace。
# args = parser.parse_args()   # 只提取已定义的参数，提取为Namespace。若有未定义参数，报错。

# print the args
print("subparser 1:")
print(args_parsed1)
print(args_unparsed1)
print("\n")

print("subparser 2:")
print(args_parsed2)
print(args_unparsed2)
print("\n")

"""
>>> input:
>>> python argparse_subparsers.py --aaa --bbb 1 2 3 4 --undefined

>>> output:
subparser 1:
Namespace(aaa=True)
['--bbb', '1', '2', '3', '4', '--undefined']
subparser 2:
Namespace(bbb=[1, 2, 3, 4])
['--aaa', '--undefined']
"""

def foo(x):
	print("\n")
	print("This is function foo:")
	print(x.bbb)

sbp2.set_defaults(func=foo)   # set_defaults 相当于是给parser对象添加一个默认参数，名为func，值为某一函数。之后调用时相当于从命名空间调用该函数，并把命名空间本身作为函数参数传入。
args_parsed2, args_punparsed2=sbp2.parse_known_args(sys.argv[1:])   # 提取已定义和未定义的参数，分别保留在第一个和第二个变量中。已定义的提取为Namespace。

print("subparser 2 after set_defaults():")
print(args_parsed2) 
args_parsed2.func(args_parsed2)

"""
>>> output:
Namespace(bbb=[1, 2, 3, 4], func=<function foo at 0x1065b1620>)

This is function foo:
[1, 2, 3, 4]
"""




