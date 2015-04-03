#pragma once

#include "function.h"

namespace ncv
{
        ///
        /// \brief create Booth function tests
        ///
        /// https://en.wikipedia.org/wiki/Test_functions_for_optimization
        ///
        NANOCV_DLL_PUBLIC std::vector<function_t> make_booth_funcs();
}
