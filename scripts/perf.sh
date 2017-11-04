#!/bin/bash

perf record --call-graph dwarf -ag -F 99 -- $@
perf report -f -g graph --no-children
