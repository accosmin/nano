#!/bin/bash

perf record --call-graph dwarf -ag -F 997 -- $@
perf report -f -g graph --no-children
