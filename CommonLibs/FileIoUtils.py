#!/usr/env/bin python
#coding=utf-8

import os
import sys

class MySentences:
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in open(self.filename, 'r'):
            yield line.strip('\r\n')