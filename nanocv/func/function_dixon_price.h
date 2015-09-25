#pragma once

#include "function.h"

namespace ncv
{
        ///
        /// \brief create Dixon-Price function tests
        ///
        NANOCV_PUBLIC functions_t make_dixon_price_funcs(size_t max_dims = 32);
}
