#pragma once

#include "average.h"
#include "softmax.h"
#include "l2nreg.hpp"

namespace nano
{
        struct average_l2n_criterion_t : public l2n_criterion_t<average_criterion_t>
        {
                NANO_MAKE_CLONABLE(average_l2n_criterion_t, "L2-norm regularized average loss")

                explicit average_l2n_criterion_t(const string_t& configuration = string_t()) :
                        l2n_criterion_t<average_criterion_t>(configuration)
                {
                }
        };

        struct softmax_l2n_criterion_t : public l2n_criterion_t<softmax_criterion_t>
        {
                NANO_MAKE_CLONABLE(softmax_l2n_criterion_t, "L2-norm regularized softmax loss, parameters: beta=5[1,10]")

                explicit softmax_l2n_criterion_t(const string_t& configuration = string_t()) :
                        l2n_criterion_t<softmax_criterion_t>(configuration)
                {
                }
        };
}
