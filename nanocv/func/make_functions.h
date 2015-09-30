#pragma once

#include "function.h"

namespace ncv
{
        ///
        /// \brief create all test functions up to the given dimension (if possible)
        ///
        NANOCV_PUBLIC functions_t make_all_test_functions(const opt_size_t max_dims);
}
