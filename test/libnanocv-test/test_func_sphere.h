#pragma once

#include "test_func.h"

namespace test
{
        ///
        /// \brief create sphere function tests
        ///
        std::vector<function_t> make_sphere_funcs(ncv::size_t max_dims = 16);
}
