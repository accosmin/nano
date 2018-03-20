#!/bin/bash

cppcheck --version
cppcheck --force --quiet --inline-suppr --enable=all --error-exitcode=1 ../src ../tests ../apps
