#pragma once

#include "function.h"

namespace ncv
{
        ///
        /// \brief create Cauchy function tests
        ///
        NANOCV_PUBLIC std::vector<function_t> make_cauchy_funcs(ncv::size_t max_dims = 16);
}
