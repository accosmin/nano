#include "sgm.h"
#include "loop.h"
#include "lrate.h"
#include "math/momentum.h"

namespace nano
{
        stoch_sgm_t::stoch_sgm_t(const string_t& configuration) :
                stoch_optimizer_t(configuration)
        {
        }

        state_t stoch_sgm_t::minimize(const stoch_params_t& param, const problem_t& problem, const vector_t& x0) const
        {
                return stoch_tune(this, param, problem, x0, make_alpha0s(), make_decays(), make_momenta());
        }

        state_t stoch_sgm_t::minimize(const stoch_params_t& param, const problem_t& problem, const vector_t& x0,
                const scalar_t alpha0, const scalar_t decay, const scalar_t momentum) const
        {
                assert(problem.size() == x0.size());

                // learning rate schedule
                lrate_t lrate(alpha0, decay);

                // first-order momentum of the update
                momentum_vector_t<vector_t> davg(momentum, x0.size());

                // optimizer
                const auto optimizer = [&] (state_t& cstate)
                {
                        // learning rate
                        const scalar_t alpha = lrate.get();

                        // descent direction
                        davg.update(-alpha * cstate.g);
                        cstate.d = davg.value();

                        // update solution
                        cstate.stoch_update(problem, 1);
                };

                // OK, assembly the optimizer
                return  stoch_loop(param, problem, x0, optimizer,
                        {{"alpha0", alpha0}, {"decay", decay}, {"momentum", momentum}});
        }
}

