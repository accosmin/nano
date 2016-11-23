#include "loop.h"
#include "adadelta.h"
#include "math/momentum.h"
#include "text/to_params.h"

namespace nano
{
        stoch_adadelta_t::stoch_adadelta_t(const string_t& configuration) :
                stoch_optimizer_t(configuration)
        {
        }

        state_t stoch_adadelta_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0) const
        {
                return stoch_tune(this, param, function, x0, make_momenta(), make_epsilons());
        }

        state_t stoch_adadelta_t::minimize(const stoch_params_t& param, const function_t& function, const vector_t& x0,
                const scalar_t momentum, const scalar_t epsilon) const
        {
                assert(function.size() == x0.size());

                // second-order momentum of the gradient
                momentum_vector_t<vector_t> gavg(momentum, x0.size());

                // second-order momentum of the step updates
                momentum_vector_t<vector_t> davg(momentum, x0.size());

                // optimizer
                const auto optimizer = [&] (state_t& cstate)
                {
                        // learning rate
                        const scalar_t alpha = 1;

                        // descent direction
                        gavg.update(cstate.g.array().square());

                        cstate.d = -cstate.g.array() *
                                   (epsilon + davg.value().array()).sqrt() /
                                   (epsilon + gavg.value().array()).sqrt();

                        davg.update(cstate.d.array().square());

                        // update solution
                        cstate.stoch_update(function, alpha);
                };

                // OK, assembly the optimizer
                return  stoch_loop(param, function, x0, optimizer,
                        to_params("momentum", momentum, "epsilon", epsilon));
        }
}

