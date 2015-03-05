#pragma once

#include "function.h"

namespace ncv
{
        ///
        /// \brief create Beale function tests
        ///
        /// https://en.wikipedia.org/wiki/Test_functions_for_optimization
        ///
        std::vector<function_t> make_beale_funcs();
}
