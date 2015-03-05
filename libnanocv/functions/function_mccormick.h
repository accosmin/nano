#pragma once

#include "function.h"

namespace ncv
{
        ///
        /// \brief create McCormick function tests
        ///
        /// https://en.wikipedia.org/wiki/Test_functions_for_optimization
        ///
        std::vector<function_t> make_mccormick_funcs();
}
