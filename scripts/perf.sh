#!/bin/bash

perf record --call-graph dwarf -- $@
perf report -g graph --no-children
