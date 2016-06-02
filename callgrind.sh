#!/bin/bash

time valgrind --dsymutil=yes --tool=callgrind --dump-instr=yes --cache-sim=yes --collect-jumps=yes $@

