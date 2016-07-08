#include "adagrad.h"
#include "loop.hpp"
#include "math/average.hpp"

namespace nano
{
        state_t stoch_adagrad_t::operator()(const stoch_params_t& param, const problem_t& problem, const vector_t& x0) const
        {
                return stoch_tune(this, param, problem, x0, make_alpha0s(), make_epsilons());
        }

        state_t stoch_adagrad_t::operator()(const stoch_params_t& param, const problem_t& problem, const vector_t& x0,
                const scalar_t alpha0, const scalar_t epsilon) const
        {
                assert(problem.size() == x0.size());

                // second-order gradient momentum
                average_vector_t<vector_t> gavg(x0.size());

                const auto op_iter = [&] (state_t& cstate)
                {
                        // learning rate
                        const scalar_t alpha = alpha0;

                        // descent direction
                        gavg.update(cstate.g.array().square());

                        cstate.d = -cstate.g.array() /
                                   (epsilon + gavg.value().array()).sqrt();

                        // update solution
                        cstate.update(problem, alpha);
                };

                // OK, assembly the optimizer
                return  stoch_loop(problem, param, state_t(problem, x0), op_iter,
                        {{"alpha0", alpha0}, {"epsilon", epsilon}});
        }
}

