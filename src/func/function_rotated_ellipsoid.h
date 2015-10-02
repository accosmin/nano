#pragma once

#include "function.h"

namespace ncv
{
        ///
        /// \brief create rotated hyper-ellipsoid function tests
        ///
        NANOCV_PUBLIC functions_t make_rotated_ellipsoid_funcs(opt_size_t max_dims = 32);
}
