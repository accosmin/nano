#pragma once

#include "function.h"

namespace ncv
{
        ///
        /// \brief create Powell function tests
        ///
        NANOCV_PUBLIC functions_t make_powell_funcs(size_t max_dims = 32);
}
