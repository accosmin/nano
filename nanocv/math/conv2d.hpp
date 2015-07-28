#pragma once

#include "conv2d_cpp.hpp"
#include "conv2d_dyn.hpp"
#include "conv2d_eig.hpp"

namespace ncv
{
        enum class conv2d_op
        {
                eig,
                cpp,
                dot,
                mad,
                dyn,
        };
}

