#pragma once

#include "regression.h"

namespace nano
{
        namespace detail
        {
                struct square_value_t
                {
                        scalar_t operator()(const vector_t& targets, const vector_t& scores) const
                        {
                                return scalar_t(0.5) * (scores - targets).array().square().sum();
                        }
                };

                struct square_vgrad_t
                {
                        vector_t operator()(const vector_t& targets, const vector_t& scores) const
                        {
                                return scores - targets;
                        }
                };
        }

        ///
        /// \brief square loss: (t - y)^2.
        ///
        using square_loss_t = regression_loss_t
        <
                detail::square_value_t,
                detail::square_vgrad_t
        >;
}

