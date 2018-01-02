#!/bin/bash

bash ci_ctest.sh
cd .. && bash <(curl -s https://codecov.io/bash)
rm -f *#*
