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
                json_reader_t& config(json_reader_t&) final;
                json_writer_t& config(json_writer_t&) const final;

        private:

                // attributes
                string_t        m_dir;  ///< directory where to load the task from
        };
}
