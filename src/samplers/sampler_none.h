#pragma once

#include "sampler.h"

namespace nano
{
        ///
        /// \brief trivial sample generator that uses the original samples as they are.
        ///
        struct sampler_none_t final : public sampler_t
        {
                explicit sampler_none_t(const string_t& configuration = string_t());

                virtual tensor3d_t input(const task_t&, const fold_t&, const size_t index) const final;
                virtual tensor3d_t target(const task_t&, const fold_t&, const size_t index) const final;
        };
}
