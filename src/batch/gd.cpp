#include "gd.h"
#include "loop.hpp"
#include "ls_init.h"
#include "ls_strategy.h"
#include "text/from_params.hpp"

namespace nano
{
        batch_gd_t::batch_gd_t(const string_t& configuration) :
                batch_optimizer_t(configuration)
        {
        }

        state_t batch_gd_t::minimize(const batch_params_t& param, const problem_t& problem, const vector_t& x0) const
        {
                assert(problem.size() == x0.size());

                // line-search initial step length
                ls_init_t ls_init(from_params<ls_initializer>(configuration(), "ls_init", ls_initializer::quadratic));

                // line-search step
                ls_strategy_t ls_step(from_params<ls_strategy>(configuration(), "ls_strat", ls_strategy::backtrack_strong_wolfe),
                        scalar_t(1e-4), scalar_t(0.1));

                const auto op = [&] (state_t& cstate, const std::size_t)
                {
                        // descent direction
                        cstate.d = -cstate.g;

                        // line-search
                        const scalar_t t0 = ls_init(cstate);
                        return ls_step(problem, t0, cstate);
                };

                // OK, assembly the optimizer
                return batch_loop(param, state_t(problem, x0), op);
        }
}

