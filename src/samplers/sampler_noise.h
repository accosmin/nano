#pragma once

#include "sampler.h"

namespace nano
{
        ///
        /// \brief generate samples by adding noise to inputs.
        ///
        struct sampler_noise_t final : public sampler_t
        {
                explicit sampler_noise_t(const string_t& configuration = string_t());

                virtual tensor3d_t input(const task_t&, const fold_t&, const size_t index) final;
                virtual tensor3d_t target(const task_t&, const fold_t&, const size_t index) final;
        };
}
