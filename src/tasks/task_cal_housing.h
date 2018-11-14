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
        class cal_housing_task_t final : public mem_csv_task_t
        {
        public:

                cal_housing_task_t() :
                        mem_csv_task_t(name(), path(), mem_csv_task_t::type::regression, target_columns())
                {
                }

                static string_t name() { return "california-housing"; }
                static string_t home() { return string_t(std::getenv("HOME")); }
                static string_t path() { return home() + "/experiments/databases/cal_housing/cal_housing.data"; }
                static indices_t target_columns() { return {size_t(8)}; }
        };
}
