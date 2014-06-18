#!/bin/bash

time valgrind --dsymutil=yes --tool=callgrind $@
        