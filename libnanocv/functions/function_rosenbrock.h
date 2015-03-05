#pragma once

#include "function.h"

namespace ncv
{
        ///
        /// \brief create Rosenbrock function tests
        ///
        /// https://en.wikipedia.org/wiki/Test_functions_for_optimization
        ///
        std::vector<function_t> make_rosenbrock_funcs();
}
