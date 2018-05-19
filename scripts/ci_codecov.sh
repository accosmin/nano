#!/bin/bash

bash ../scripts/ci_ctest.sh
bash <(curl -s https://codecov.io/bash) -R .. -g "**/apps/**" -g "**/tests/**"
rm -f *#*
