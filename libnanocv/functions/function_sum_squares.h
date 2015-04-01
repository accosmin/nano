#pragma once

#include "function.h"

namespace ncv
{
        ///
        /// \brief create sum of squares function tests
        ///
        std::vector<function_t> make_sum_squares_funcs(ncv::size_t max_dims = 32);
}
