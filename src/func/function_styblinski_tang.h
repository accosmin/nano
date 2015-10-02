#pragma once

#include "function.h"

namespace ncv
{
        ///
        /// \brief create Styblinski-Tang function tests
        ///
        NANOCV_PUBLIC functions_t make_styblinski_tang_funcs(opt_size_t max_dims = 16);
}
