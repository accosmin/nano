#pragma once

#include "function.h"

namespace ncv
{
        ///
        /// \brief create Goldstein-Price function tests
        ///
        /// https://en.wikipedia.org/wiki/Test_functions_for_optimization
        ///
        NANOCV_PUBLIC std::vector<function_t> make_goldstein_price_funcs();
}
