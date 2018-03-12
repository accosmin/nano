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

                iris_task_t();
                bool populate() override;
                void to_json(json_t&) const final;
                void from_json(const json_t&) final;

        private:

                // attributes
                string_t        m_dir;  ///< directory where to load the task from
        };
}
