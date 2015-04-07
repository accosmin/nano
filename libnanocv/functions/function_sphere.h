#pragma once

#include "function.h"

namespace ncv
{
        ///
        /// \brief create sphere function tests
        ///
        NANOCV_PUBLIC std::vector<function_t> make_sphere_funcs(ncv::size_t max_dims = 16);
}
