#pragma once

#include "function.h"

namespace ncv
{
        ///
        /// \brief create Himmelblau function tests
        ///
        /// https://en.wikipedia.org/wiki/Test_functions_for_optimization
        ///
        NANOCV_PUBLIC functions_t make_himmelblau_funcs();
}
