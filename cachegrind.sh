#!/bin/bash

time valgrind --tool=cachegrind --dump-instr=yes --collect-jumps=yes $@
        
