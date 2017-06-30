#pragma once

#include "solver_stoch.h"

namespace nano
{
        ///
        /// \brief stochastic RMSProp (AdaGrad with an exponentially weighted running average of the gradients)
        ///     see Lecture 6.5 - rmsprop, COURSERA: Neural Networks for Machine Learning
        ///      by Tieleman, T. and Hinton, G. (2012)
        ///
        struct stoch_rmsprop_t final : public stoch_solver_t
        {
                explicit stoch_rmsprop_t(const string_t& params = string_t());

                function_state_t minimize(const stoch_params_t&, const function_t&, const vector_t& x0) const override;

                static function_state_t minimize(const stoch_params_t&, const function_t&, const vector_t& x0,
                        const scalar_t alpha0, const scalar_t decay, const scalar_t momentum, const scalar_t epsilon);
        };
}
