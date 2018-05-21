#pragma once

#include "task_mem_tensor.h"

namespace nano
{
        ///
        ///
        /// \brief synthetic task for:
        ///     - the regression problem of fitting a noisy affine transformation:
        ///             y = A * x + b + noise
        ///
        /// parameters:
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
                void to_json(json_t&) const final;
                void from_json(const json_t&) final;

        private:

                // attributes
                tensor_size_t           m_isize{32};
                tensor_size_t           m_osize{32};
                size_t                  m_folds{10};
                size_t                  m_count{1024};
                scalar_t                m_noise{static_cast<scalar_t>(1e-3)};
        };
}
