#pragma once

#include "function.h"

namespace ncv
{
        ///
        /// \brief create Dixon-Price function tests
        ///
        std::vector<function_t> make_dixon_price_funcs(ncv::size_t max_dims = 32);
}
