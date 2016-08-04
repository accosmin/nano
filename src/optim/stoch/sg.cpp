#include "sg.h"
#include "lrate.h"
#include "loop.hpp"

namespace nano
{
        stoch_sg_t::stoch_sg_t(const string_t& configuration) :
                stoch_optimizer_t(configuration)
        {
        }

        state_t stoch_sg_t::minimize(const stoch_params_t& param, const problem_t& problem, const vector_t& x0) const
        {
                return stoch_tune(this, param, problem, x0, make_alpha0s(), make_decays());
        }

        state_t stoch_sg_t::minimize(const stoch_params_t& param, const problem_t& problem, const vector_t& x0,
                const scalar_t alpha0, const scalar_t decay) const
        {
                assert(problem.size() == x0.size());

                // learning rate schedule
                lrate_t lrate(alpha0, decay);

                const auto op_iter = [&] (state_t& cstate)
                {
                        // learning rate
                        const scalar_t alpha = lrate.get();

                        // descent direction
                        cstate.d = -cstate.g;

                        // update solution
                        cstate.update(problem, alpha);
                };

                // OK, assembly the optimizer
                return  stoch_loop(problem, param, state_t(problem, x0), op_iter,
                        {{"alpha0", alpha0}, {"decay", decay}});
        }
}

