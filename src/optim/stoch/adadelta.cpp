#include "adadelta.h"
#include "loop.hpp"
#include "math/momentum.hpp"

namespace nano
{
        stoch_adadelta_t::stoch_adadelta_t(const string_t& configuration) :
                stoch_optimizer_t(configuration)
        {
        }

        state_t stoch_adadelta_t::minimize(const stoch_params_t& param, const problem_t& problem, const vector_t& x0) const
        {
                return stoch_tune(this, param, problem, x0, make_momenta(), make_epsilons());
        }

        state_t stoch_adadelta_t::minimize(const stoch_params_t& param, const problem_t& problem, const vector_t& x0,
                const scalar_t momentum, const scalar_t epsilon) const
        {
                assert(problem.size() == x0.size());

                // second-order momentum of the gradient
                momentum_vector_t<vector_t> gavg(momentum, x0.size());

                // second-order momentum of the step updates
                momentum_vector_t<vector_t> davg(momentum, x0.size());

                const auto op_iter = [&] (state_t& cstate)
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
                        cstate.update(problem, alpha);
                };

                // OK, assembly the optimizer
                return  stoch_loop(problem, param, state_t(problem, x0), op_iter,
                        {{"momentum", momentum}, {"epsilon", epsilon}});
        }
}

