#include "gd.h"
#include "loop.h"
#include "ls_init.h"
#include "ls_strategy.h"

namespace nano
{
        batch_gd_t::batch_gd_t(const string_t& configuration) :
                batch_solver_t(to_params(configuration,
                "ls_init", ls_initializer::quadratic,
                "ls_strat", ls_strategy::backtrack_strong_wolfe,
                "c1", 1e-4, "c2", 0.1))
        {
        }

        function_state_t batch_gd_t::minimize(const batch_params_t& param, const function_t& function, const vector_t& x0) const
        {
                return  minimize(param, function, x0,
                        from_params<ls_initializer>(config(), "ls_init"),
                        from_params<ls_strategy>(config(), "ls_strat"),
                        from_params<scalar_t>(config(), "c1"),
                        from_params<scalar_t>(config(), "c2"));
        }

        function_state_t batch_gd_t::minimize(const batch_params_t& param, const function_t& function, const vector_t& x0,
                const ls_initializer linit, const ls_strategy lstrat, const scalar_t c1, const scalar_t c2) const
        {
                // line-search initial step length
                ls_init_t ls_init(linit);

                // line-search step
                ls_strategy_t ls_step(lstrat, c1, c2);

                const auto op = [&] (function_state_t& cstate, const std::size_t)
                {
                        // descent direction
                        cstate.d = -cstate.g;

                        // line-search
                        const scalar_t t0 = ls_init(cstate);
                        return ls_step(function, t0, cstate);
                };

                // OK, assembly the optimizer
                return batch_loop(param, function, x0, op);
        }
}
