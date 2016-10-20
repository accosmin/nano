#include "gd.h"
#include "loop.h"
#include "ls_init.h"
#include "ls_strategy.h"
#include "text/to_params.h"
#include "text/from_params.h"

namespace nano
{
        batch_gd_t::batch_gd_t(const string_t& configuration) :
                batch_optimizer_t(concat_params(configuration, "ls_init=init-quadratic,ls_strat=backtrack-strong-wolfe,c1=1e-4,c2=0.1"))
        {
        }

        rbatch_optimizer_t batch_gd_t::clone() const
        {
                return std::make_unique<batch_gd_t>(*this);
        }

        state_t batch_gd_t::minimize(const batch_params_t& param, const problem_t& problem, const vector_t& x0) const
        {
                assert(problem.size() == x0.size());

                // line-search initial step length
                ls_init_t ls_init(from_params<ls_initializer>(config(), "ls_init"));

                // line-search step
                const auto c1 = from_params<scalar_t>(config(), "c1");
                const auto c2 = from_params<scalar_t>(config(), "c2");
                ls_strategy_t ls_step(from_params<ls_strategy>(config(), "ls_strat"), c1, c2);

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

