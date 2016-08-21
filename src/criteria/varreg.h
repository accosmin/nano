#pragma once

#include "average.h"
#include "softmax.h"
#include "varreg.hpp"

namespace nano
{
        struct average_var_criterion_t : public var_criterion_t<average_criterion_t>
        {
                NANO_MAKE_CLONABLE(average_var_criterion_t, "variance-regularized average loss", "")

                explicit average_var_criterion_t(const string_t& configuration = string_t()) :
                        var_criterion_t<average_criterion_t>(configuration)
                {
                }
        };

        struct softmax_var_criterion_t : public var_criterion_t<softmax_criterion_t>
        {
                NANO_MAKE_CLONABLE(softmax_var_criterion_t, "variance-regularized softmax loss", "beta=5[1,10]")

                explicit softmax_var_criterion_t(const string_t& configuration = string_t()) :
                        var_criterion_t<softmax_criterion_t>(configuration)
                {
                }
        };
}

