#pragma once

#include "task_mem_csv.h"

namespace nano
{
        ///
        /// WINE task:
        ///      - wine quality classification
        ///      - 13 attributes
        ///      - 3 classes
        ///
        /// http://archive.ics.uci.edu/ml/datasets/Wine
        ///
        class wine_task_t final : public mem_csv_task_t
        {
        public:

                wine_task_t() :
                        mem_csv_task_t(name(), path(), mem_csv_task_t::type::classification, target_columns())
                {
                }

                static string_t name() { return "WINE"; }
                static string_t home() { return string_t(std::getenv("HOME")); }
                static string_t path() { return home() + "/experiments/databases/wine/wine.data"; }
                static indices_t target_columns() { return {size_t(0)}; }
        };
}
