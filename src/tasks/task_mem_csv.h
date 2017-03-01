#pragma once

#include "task_mem_tensor.h"

namespace nano
{
        ///
        /// in-memory task with the data loaded from csv:
        ///     - assumes the target is one dimensional (either a class or a scalar to predict)
        ///     - assumes no missing data
        ///
        struct mem_csv_task_t : public mem_tensor_task_t
        {
                mem_csv_task_t(
                        const dim3d_t& idims,
                        const dim3d_t& odims,
                        const size_t fsize,
                        const string_t& configuration = string_t()) :
                        mem_tensor_task_t(idims, odims, fsize, configuration)
                {
                }

                ///
                /// \brief load CSV for classification
                ///
                bool load_classification(const string_t& path, const string_t& task_name,
                        const size_t expected_samples,
                        const scalars_t& scales, const strings_t& labels, const size_t label_column);

                ///
                /// \brief load CSV for regression
                ///
                bool load_regression(const string_t& path, const string_t& task_name,
                        const size_t expected_samples,
                        const scalars_t& scales, const size_t target_column);
        };
}
