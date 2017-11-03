#!/bin/bash

bash ci_test.sh
cd .. && bash <(curl -s https://codecov.io/bash)
rm -f *#*
