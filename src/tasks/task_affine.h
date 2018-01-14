#pragma once

#include "task_mem_tensor.h"

namespace nano
{
        ///
        /// \brief
        ///
        enum class affine_task_type
        {
                regression,
                classification
        };

        template <>
        inline enum_map_t<affine_task_type> enum_string<affine_task_type>()
        {
                return
                {
                        { affine_task_type::regression,         "regression" },
                        { affine_task_type::classification,     "classification" }
                };
        }

        ///
        ///
        /// \brief synthetic task for:
        ///     - the regression problem of fitting a noisy affine transformation:
        ///             y = A * x + b + noise
        ///     - the classification problem of predicting the sign of a noisy affine transformation:
        ///             y = sign(A * x + b + noise)
        ///
        /// parameters:
        ///     type    - regression/classification
        ///     isize   - input dimensions
        ///     osize   - output dimensions
        ///     noise   - additive noise sampled uniformly from [-noise,+noise]
        ///     count   - number of samples (training + validation + test)
        ///
        class affine_task_t final : public mem_tensor_task_t
        {
        public:

                affine_task_t();
                bool populate() override;
                json_reader_t& config(json_reader_t&) final;
                json_writer_t& config(json_writer_t&) const final;

        private:

                // attributes
                affine_task_type        m_type{affine_task_type::regression};
                tensor_size_t           m_isize{32};
                tensor_size_t           m_osize{32};
                size_t                  m_count{1024};
                scalar_t                m_noise{static_cast<scalar_t>(1e-3)};
        };
}
