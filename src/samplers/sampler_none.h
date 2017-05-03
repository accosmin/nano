#pragma once

#include "sampler.h"

namespace nano
{
        ///
        /// \brief trivial sample generator that uses the original samples as they are.
        ///
        struct sampler_none_t final : public sampler_t
        {
                explicit sampler_none_t(const string_t& configuration = string_t()) :
                        sampler_t(configuration) {}

                virtual void get(tensor3d_t&, vector_t*, string_t*) final {}
        };
}
