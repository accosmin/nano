#pragma once

#include "nanocv/arch.h"
#include "nanocv/tensor.h"

namespace ncv
{
        class model_t;

        ///
        /// \brief construct (from a random initialization) an input that matches closely the given target
        ///
        NANOCV_PUBLIC tensor_t generate_match_target(const model_t& model, const vector_t& target);
}

