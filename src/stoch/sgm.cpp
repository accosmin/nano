#include "sgm.h"
#include "loop.hpp"
#include "math/momentum.hpp"

namespace nano
{
        stoch_sgm_t::stoch_sgm_t(const string_t& configuration) :
                stoch_optimizer_t(configuration)
        {
        }

        state_t stoch_sgm_t::minimize(const stoch_params_t& param, const problem_t& problem, const vector_t& x0) const
        {
                return stoch_tune(this, param, problem, x0, make_alpha0s(), make_momenta());
        }

        state_t stoch_sgm_t::minimize(const stoch_params_t& param, const problem_t& problem, const vector_t& x0,
                const scalar_t alpha0, const scalar_t momentum) const
        {
                assert(problem.size() == x0.size());

                // first-order momentum of the update
                momentum_vector_t<vector_t> davg(momentum, x0.size());

                const auto op_iter = [&] (state_t& cstate)
                {
                        // learning rate
                        const scalar_t alpha = alpha0;

                        // descent direction
                        davg.update(-alpha * cstate.g);
                        cstate.d = davg.value();

                        // update solution
                        cstate.update(problem, 1);
                };

                // OK, assembly the optimizer
                return  stoch_loop(problem, param, state_t(problem, x0), op_iter,
                        {{"alpha0", alpha0}, {"momentum", momentum}});
        }
}

