#pragma once

#include "corr2d_cpp.hpp"
#include "corr2d_dyn.hpp"
#include "corr2d_egb.hpp"
#include "corr2d_egr.hpp"

namespace ncv
{
        enum class corr2d_op
        {
                egb,
                egr,
                cpp,
                mdk,
                mdo,
                dyn
        };
}


