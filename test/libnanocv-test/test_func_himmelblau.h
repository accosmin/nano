#pragma once

#include "test_func.h"

namespace test
{
        ///
        /// \brief create Himmelblau function tests
        ///
        /// https://en.wikipedia.org/wiki/Test_functions_for_optimization
        ///
        std::vector<function_t> make_himmelblau_funcs();
}
