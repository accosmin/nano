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
                        mem_csv_task_t(name(), path(), label_column())
                {
                }

                static string_t name() { return "WINE"; }
                static string_t path() { return string_t(std::getenv("HOME")) + "/experiments/databases/wine/wine.data"; }
                static size_t label_column() { return 0; }
        };
}
