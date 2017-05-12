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
        struct wine_task_t final : public mem_csv_task_t
        {
                explicit wine_task_t(const string_t& configuration = string_t());

                virtual bool populate() override;
        };
}
