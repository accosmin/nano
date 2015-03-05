#pragma once

#include "function.h"

namespace test
{
        ///
        /// \brief create ellipse function tests
        ///
        std::vector<function_t> make_ellipse_funcs(ncv::size_t max_dims = 32);
}
