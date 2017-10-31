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
        struct iris_task_t final : public mem_csv_task_t
        {
                explicit iris_task_t(const string_t& params = string_t());

                bool populate() override;
        };
}
