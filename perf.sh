#!/bin/bash

perf record --call-graph dwarf -ag -F 997 -- $@
perf report -g graph --no-children
