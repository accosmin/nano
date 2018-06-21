#include "solver_gd.h"

using namespace nano;

tuner_t solver_gd_t::tuner() const
{
        tuner_t tuner;
        tuner.add_finite("c1", 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1);
        tuner.add_finite("c2", 0.2, 0.5, 0.9);
        tuner.add_enum<lsearch_t::initializer>("ls_init");
        tuner.add_enum<lsearch_t::strategy>("ls_strat");
        return tuner;
}

void solver_gd_t::from_json(const json_t& json)
{
        nano::from_json(json, "ls_init", m_ls_init, "ls_strat", m_ls_strat, "c1", m_c1, "c2", m_c2);
}

void solver_gd_t::to_json(json_t& json) const
{
        nano::to_json(json,
                "ls_init", m_ls_init, "ls_inits", join(enum_values<lsearch_t::initializer>()),
                "ls_strat", m_ls_strat, "ls_strats", join(enum_values<lsearch_t::strategy>()),
                "c1", m_c1, "c2", m_c2);
}

solver_state_t solver_gd_t::minimize(const size_t max_iterations, const scalar_t epsilon,
        const function_t& function, const vector_t& x0, const logger_t& logger) const
{
        lsearch_t lsearch(m_ls_init, m_ls_strat, m_c1, m_c2);

        const auto op = [&] (solver_state_t& cstate, const std::size_t)
        {
                // descent direction
                cstate.d = -cstate.g;

                // line-search
                return lsearch(function, cstate);
        };

        // assembly the solver
        return loop(function, x0, max_iterations, epsilon, logger, op);
}
