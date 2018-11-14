#pragma once

#include "task_mem_tensor.h"

namespace nano
{
        class table_t;

        ///
        /// \brief in-memory task with the data loaded from csv:
        ///     - assumes the target is the concatenation of several columns for regression
        ///     - assumes the target is the a column for classification
        ///     - assumes no missing data
        ///
        class mem_csv_task_t : public mem_tensor_task_t
        {
        public:

                enum class type
                {
                        regression,
                        classification
                };

                mem_csv_task_t(string_t name, string_t path, const type, indices_t target_columns);

                bool populate() final;
                void to_json(json_t&) const final;
                void from_json(const json_t&) final;

        private:

                bool is_target(const size_t col) const
                {
                        return  std::find(m_target_columns.begin(), m_target_columns.end(), col) !=
                                m_target_columns.end();
                }

                void reconfig(const tensor_size_t n_attributes, const tensor_size_t n_outputs)
                {
                        mem_tensor_task_t::reconfig(
                                make_dims(n_attributes, 1, 1),
                                make_dims(n_outputs, 1, 1),
                                m_folds);
                }

                bool populate_regression(const table_t&);
                bool populate_classification(const table_t&);
                bool populate(const tensor3ds_t& inputs, const tensor3ds_t& targets, const strings_t& labels);

        private:

                // attributes
                string_t                m_name;                 ///< task name
                string_t                m_path;                 ///< path where to load the task from
                type                    m_type;                 ///< task type
                indices_t               m_target_columns;       ///< column indices to construct the target
                size_t                  m_folds{10};            ///< #folds
                int                     m_train_percentage{40}; ///< percentage of training samples / fold
                int                     m_valid_percentage{30}; ///< percentage of validation samples / fold
        };
}
