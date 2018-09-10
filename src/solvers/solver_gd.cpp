#include "solver_gd.h"

using namespace nano;

tuner_t solver_gd_t::tuner() const
{
        tuner_t tuner;
        tuner.add_enum<lsearch_t::initializer>("init");
        tuner.add_enum<lsearch_t::strategy>("strat");
        return tuner;
}

void solver_gd_t::from_json(const json_t& json)
{
        nano::from_json(json, "init", m_init, "strat", m_strat, "c1", m_c1, "c2", m_c2);
}

void solver_gd_t::to_json(json_t& json) const
{
        nano::to_json(json,
                "init", m_init, "inits", join(enum_values<lsearch_t::initializer>()),
                "strat", m_strat, "strats", join(enum_values<lsearch_t::strategy>()),
                "c1", m_c1, "c2", m_c2);
}

solver_state_t solver_gd_t::minimize(const size_t max_iterations, const scalar_t epsilon,
        const function_t& f, const vector_t& x0, const logger_t& logger) const
{
        lsearch_t lsearch(m_init, m_strat, m_c1, m_c2);

        const auto op = [&] (const function_t& function, solver_state_t& cstate, const size_t)
        {
                // descent direction
                cstate.d = -cstate.g;

                // line-search
                return lsearch(function, cstate);
        };

        // assembly the solver
        return loop(f, x0, max_iterations, epsilon, logger, op);
}
