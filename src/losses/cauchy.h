#pragma once

#include "regression.h"

namespace nano
{
        namespace detail
        {
                struct cauchy_value_t
                {
                        scalar_t operator()(const vector_t& targets, const vector_t& scores) const
                        {
                                return ((targets - scores).array().square() + 1).log().sum();
                        }
                };

                struct cauchy_vgrad_t
                {
                        vector_t operator()(const vector_t& targets, const vector_t& scores) const
                        {
                                return 2 * (scores - targets).array() / (1 + (scores - targets).array().square());
                        }
                };
        }

        ///
        /// \brief robust-to-noise Cauchy loss: log(1 + (t - y)^2).
        ///
        using cauchy_loss_t = regression_loss_t
        <
                detail::cauchy_value_t,
                detail::cauchy_vgrad_t
        >;
}

