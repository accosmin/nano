#pragma once

#include "test_func.h"

namespace test
{
        ///
        /// \brief create Matyas function tests
        ///
        /// https://en.wikipedia.org/wiki/Test_functions_for_optimization
        ///
        std::vector<function_t> make_matyas_funcs();
}
