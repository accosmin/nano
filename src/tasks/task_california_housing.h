#pragma once

#include "task_mem_csv.h"

namespace nano
{
        ///
        /// California housing task:
        ///     - predict the median house value
        ///     - 9 attributes
        ///
        /// http://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html
        ///
        class california_housing_task_t final : public mem_csv_task_t
        {
        public:

                california_housing_task_t() :
                        mem_csv_task_t(name(), path(), label_column())
                {
                }

                static string_t name() { return "california-housing"; }
                static string_t path() { return string_t(std::getenv("HOME")) + "/experiments/databases/california_housing/cal_housing.data"; }
                static size_t label_column() { return 8; }
        };
}
