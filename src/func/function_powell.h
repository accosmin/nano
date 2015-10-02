#pragma once

#include "function.h"

namespace ncv
{
        ///
        /// \brief create Powell function tests
        ///
        NANOCV_PUBLIC functions_t make_powell_funcs(opt_size_t max_dims = 32);
}
