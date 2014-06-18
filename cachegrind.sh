#!/bin/bash

time valgrind --tool=cachegrind $@
        