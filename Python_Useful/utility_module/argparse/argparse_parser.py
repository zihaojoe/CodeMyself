#!/usr/bin/env python
import argparse
import sys

parser = argparse.ArgumentParser(description="This is a demo of the argparse")
parser.add_argument("--bool", "-b", default=False, action="store_true",
                    help="This is the description of the paramter. If given, the value of the relative variable is True")
parser.add_argument("--number", "-n", type=int, nargs='+',
                    help="This is the description of the paramter. narg='+', '*','?' or N")

args_parsed, args_unparsed=parser.parse_known_args(sys.argv[1:])   # 提取已定义和未定义的参数，分别保留在第一个和第二个变量中。已定义的提取为Namespace。
# args = parser.parse_args()   # 只提取已定义的参数，提取为Namespace。若有未定义参数，报错。

# print the args
print(args_parsed)
print(args_parsed.bool)
print(args_parsed.number)
print(args_unparsed)

"""
>>> python argparse_parser.py --bool --number 1 2 3 4 --undefined

>>> output:
Namespace(bool=True, number=[1, 2, 3, 4])
True
[1, 2, 3, 4]
['--undefined']
"""
