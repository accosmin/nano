#pragma once

#include "function.h"

namespace ncv
{
        ///
        /// \brief create hyper-ellipsoid function tests
        ///
        std::vector<function_t> make_ellipsoid_funcs(ncv::size_t max_dims = 32);
}
