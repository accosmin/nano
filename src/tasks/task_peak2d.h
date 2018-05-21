#pragma once

#include "task_mem_tensor.h"

namespace nano
{
        ///
        ///
        /// \brief synthetic task for:
        ///     - the regression problem of predicting the (relative) location of peak (px, py) in a noisy 2D image:
        ///             y = (px, py)
        ///
        /// parameters:
        ///     type    - regression/classification
        ///     irows   - number of rows of the input image
        ///     icols   - number of columns of the input image
        ///     noise   - additive noise sampled uniformly from [-noise,+noise]
        ///     count   - number of samples (training + validation + test)
        ///
        class peak2d_task_t final : public mem_tensor_task_t
        {
        public:

                peak2d_task_t();
                bool populate() override;
                void to_json(json_t&) const final;
                void from_json(const json_t&) final;

        private:

                // attributes
                tensor_size_t           m_irows{32};
                tensor_size_t           m_icols{32};
                size_t                  m_folds{10};
                size_t                  m_count{1024};
                scalar_t                m_noise{static_cast<scalar_t>(1e-3)};
        };
}
