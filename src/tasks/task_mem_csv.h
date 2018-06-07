#pragma once

#include "task_mem_tensor.h"

namespace nano
{
        ///
        /// \brief in-memory task with the data loaded from csv:
        ///     - assumes the target is one dimensional (either a class or a scalar to predict)
        ///     - assumes no missing data
        ///
        class mem_csv_task_t : public mem_tensor_task_t
        {
        public:

                mem_csv_task_t(string_t name, string_t path, const size_t label_column);

                bool populate() final;
                void to_json(json_t&) const final;
                void from_json(const json_t&) final;

        private:

                // attributes
                string_t        m_name;                 ///< task name
                string_t        m_path;                 ///< path where to load the task from
                size_t          m_label_column{0};      ///< index of the label column
                size_t          m_folds{10};            ///< #folds
                int             m_train_percentage{40}; ///< percentage of training samples / fold
                int             m_valid_percentage{30}; ///< percentage of validation samples / fold
        };
}
