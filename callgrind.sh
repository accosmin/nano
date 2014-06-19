#!/bin/bash

time valgrind --dsymutil=yes --tool=callgrind --dump-instr=yes --collect-jumps=yes $@
        
