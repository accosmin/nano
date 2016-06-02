#!/bin/bash

time valgrind --tool=memcheck --track-origins=yes --leak-check=full $@
        
