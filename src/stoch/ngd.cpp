#include "ngd.h"
#include "loop.h"
#include "lrate.h"

namespace nano
{
        stoch_ngd_t::stoch_ngd_t(const string_t& configuration) :
                stoch_optimizer_t(configuration)
        {
        }

        state_t stoch_ngd_t::minimize(const stoch_params_t& param, const problem_t& problem, const vector_t& x0) const
        {
                return stoch_tune(this, param, problem, x0, make_alpha0s(), make_decays());
        }

        state_t stoch_ngd_t::minimize(const stoch_params_t& param, const problem_t& problem, const vector_t& x0,
                const scalar_t alpha0, const scalar_t decay) const
        {
                assert(problem.size() == x0.size());

                // initial state
                state_t istate(problem.size());
                istate.stoch_update(problem, x0);

                // learning rate schedule
                lrate_t lrate(alpha0, decay, param.m_epoch_size);

                // optimizer
                const auto optimizer = [&] (state_t& cstate)
                {
                        // learning rate
                        const scalar_t alpha = lrate.get();

                        // descent direction
                        const scalar_t norm = 1 / cstate.g.template lpNorm<2>();
                        cstate.d = -cstate.g * norm;

                        // update solution
                        cstate.stoch_update(problem, alpha);
                };

                // OK, assembly the optimizer
                return  stoch_loop(param, problem, istate, optimizer,
                        {{"alpha0", alpha0}, {"decay", decay}});
        }
}

