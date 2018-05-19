#!/bin/bash

bash ../scripts/ci_ctest.sh
bash <(curl -s https://codecov.io/bash) -R .. -F "src/*"
rm -f *#*
