#pragma once

#include "regression.h"

namespace nano
{
        ///
        /// \brief robust-to-noise Cauchy loss: log(1 + (target - score)^2).
        ///
        namespace detail
        {
                struct cauchy_t
                {
                        static scalar_t value(const vector_t& targets, const vector_t& scores)
                        {
                                return ((targets - scores).array().square() + 1).log().sum();
                        }

                        static vector_t vgrad(const vector_t& targets, const vector_t& scores)
                        {
                                return 2 * (scores - targets).array() / (1 + (scores - targets).array().square());
                        }
                };
        }

        using cauchy_loss_t = regression_t<detail::cauchy_t>;
}

