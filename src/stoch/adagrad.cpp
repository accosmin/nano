#include "loop.h"
#include "adagrad.h"

namespace nano
{
        stoch_adagrad_t::stoch_adagrad_t(const string_t& configuration) :
                stoch_optimizer_t(configuration)
        {
        }

        state_t stoch_adagrad_t::minimize(const stoch_params_t& param, const problem_t& problem, const vector_t& x0) const
        {
                return stoch_tune(this, param, problem, x0, make_alpha0s(), make_epsilons());
        }

        state_t stoch_adagrad_t::minimize(const stoch_params_t& param, const problem_t& problem, const vector_t& x0,
                const scalar_t alpha0, const scalar_t epsilon) const
        {
                assert(problem.size() == x0.size());

                // initial state
                state_t istate(problem.size());
                istate.stoch_update(problem, x0);

                // second-order gradient momentum
                vector_t gsum2 = vector_t::Zero(x0.size());

                const auto op_iter = [&] (state_t& cstate)
                {
                        // learning rate
                        const scalar_t alpha = alpha0;

                        // descent direction
                        gsum2.array() += cstate.g.array().square();

                        cstate.d = -cstate.g.array() / (epsilon + gsum2.array()).sqrt();

                        // update solution
                        cstate.stoch_update(problem, alpha);
                };

                // OK, assembly the optimizer
                return  stoch_loop(param, istate, op_iter,
                        {{"alpha0", alpha0}, {"epsilon", epsilon}});
        }
}

