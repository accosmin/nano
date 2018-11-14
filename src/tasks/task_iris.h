#pragma once

#include "task_mem_csv.h"

namespace nano
{
        ///
        /// IRIS task:
        ///     - irises (the plant) classification
        ///     - 4 attributes
        ///     - 3 classes
        ///
        /// http://archive.ics.uci.edu/ml/datasets/Iris
        ///
        class iris_task_t final : public mem_csv_task_t
        {
        public:

                iris_task_t() :
                        mem_csv_task_t(name(), path(), mem_csv_task_t::type::classification, target_columns())
                {
                }

                static string_t name() { return "IRIS"; }
                static string_t home() { return string_t(std::getenv("HOME")); }
                static string_t path() { return home() + "/experiments/databases/iris/iris.data"; }
                static indices_t target_columns() { return {size_t(4)}; }
        };
}
