#!/usr/bin/env python3
import os
import requests
import torch
import argparse

def tstring(astring):
    return astring


parser = argparse.ArgumentParser()
# subparsers = parser.add_subparsers(help='for dealing with the command')

parser.add_argument("-g", "--gpus", type=int, default=0,
                    help="number of gpus")

parser.add_argument("-d", "--delay", type=int, default=0,
                    help="delay for running next task")

# parser.add_argument("-c", type=tstring, nargs="+",
#                     help="number of pgus")

# a_parser = subparsers.add_parser("*")
# parser.add_argument('command', nargs = "*", help = 'Other commands')

args, rest = parser.parse_known_args()
command = " ".join(rest)

# print(a_parser.parse_args())
url = "http://127.0.0.1:18812/update_process"
myobj = {'path': os.getcwd(), "command": command, "gpus": args.gpus, "delay": args.delay}
x = requests.post(url, data = myobj)

print(x.text)