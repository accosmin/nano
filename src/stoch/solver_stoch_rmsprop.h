#pragma once

#include "solver_stoch.h"

namespace nano
{
        ///
        /// \brief stochastic RMSProp (AdaGrad with an exponentially weighted running average of the gradients)
        ///     see Lecture 6.5 - rmsprop, COURSERA: Neural Networks for Machine Learning
        ///      by Tieleman, T. and Hinton, G. (2012)
        ///
        class stoch_rmsprop_t final : public stoch_solver_t
        {
        public:

                solver_state_t minimize(const stoch_params_t&, const function_t&, const vector_t& x0) const override;

                static solver_state_t minimize(const stoch_params_t&, const function_t&, const vector_t& x0,
                        const scalar_t alpha0, const scalar_t momentum, const scalar_t epsilon);
        };
}
