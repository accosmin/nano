#include "ls_init.h"
#include "solver_gd.h"
#include "ls_strategy.h"

using namespace nano;

void solver_gd_t::from_json(const json_t& json)
{
        nano::from_json(json, "ls_init", m_ls_init, "ls_strat", m_ls_strat, "c1", m_c1, "c2", m_c2);
}

void solver_gd_t::to_json(json_t& json) const
{
        nano::to_json(json,
                "ls_init", m_ls_init, "ls_inits", join(enum_values<ls_initializer>()),
                "ls_strat", m_ls_strat, "ls_strats", join(enum_values<ls_strategy>()),
                "c1", m_c1, "c2", m_c2);
}

solver_state_t solver_gd_t::minimize(const batch_params_t& param, const function_t& function, const vector_t& x0) const
{
        // line-search initial step length
        ls_init_t ls_init(m_ls_init);

        // line-search step
        ls_strategy_t ls_step(m_ls_strat, m_c1, m_c2);

        const auto op = [&] (solver_state_t& cstate, const std::size_t)
        {
                // descent direction
                cstate.d = -cstate.g;

                // line-search
                const scalar_t t0 = ls_init(cstate);
                return ls_step(function, t0, cstate);
        };

        // assembly the solver
        return loop(param, function, x0, op);
}
