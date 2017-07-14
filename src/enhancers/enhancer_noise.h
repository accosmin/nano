#pragma once

#include "enhancer.h"

namespace nano
{
        ///
        /// \brief generate samples by adding salt & pepper noise to inputs.
        ///
        struct enhancer_noise_t final : public enhancer_t
        {
                explicit enhancer_noise_t(const string_t& params = string_t());

                virtual sample_t get(const task_t&, const fold_t&, const size_t index) const final;
        };
}
