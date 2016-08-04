#include "ngd.h"
#include "loop.hpp"

namespace nano
{
        stoch_ngd_t::stoch_ngd_t(const string_t& configuration) :
                stoch_optimizer_t(configuration)
        {
        }

        state_t stoch_ngd_t::minimize(const stoch_params_t& param, const problem_t& problem, const vector_t& x0) const
        {
                return stoch_tune(this, param, problem, x0, make_alpha0s());
        }

        state_t stoch_ngd_t::minimize(const stoch_params_t& param, const problem_t& problem, const vector_t& x0,
                const scalar_t alpha0) const
        {
                assert(problem.size() == x0.size());

                const auto op_iter = [&] (state_t& cstate)
                {
                        // learning rate
                        const scalar_t alpha = alpha0;

                        // descent direction
                        const scalar_t norm = 1 / cstate.g.template lpNorm<2>();
                        cstate.d = -cstate.g * norm;

                        // update solution
                        cstate.update(problem, alpha);
                };

                // OK, assembly the optimizer
                return  stoch_loop(problem, param, state_t(problem, x0), op_iter,
                        {{"alpha0", alpha0}});
        }
}

