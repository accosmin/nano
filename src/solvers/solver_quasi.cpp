#include "solver_quasi.h"
#include <deque>

using namespace nano;

tuner_t solver_quasi_t::tuner() const
{
        tuner_t tuner;
        tuner.add_enum<lsearch_t::initializer>("init");
        tuner.add_enum<lsearch_t::strategy>("strat");
        return tuner;
}

void solver_quasi_t::from_json(const json_t& json)
{
        nano::from_json(json,
                "init", m_init, "strat", m_strat,
                "c1", m_c1, "c2", m_c2);
}

void solver_quasi_t::to_json(json_t& json) const
{
        nano::to_json(json,
                "init", m_init, "inits", join(enum_values<lsearch_t::initializer>()),
                "strat", m_strat, "strats", join(enum_values<lsearch_t::strategy>()),
                "c1", m_c1, "c2", m_c2);
}

solver_state_t solver_quasi_t::minimize(const size_t max_iterations, const scalar_t epsilon,
        const function_t& function, const vector_t& x0, const logger_t& logger) const
{
        lsearch_t lsearch(m_init, m_strat, m_c1, m_c2);

        // previous state
        solver_state_t pstate(function.size());

        // current approximation of the Hessian
        matrix_t I = matrix_t::Identity(function.size(), function.size());
        matrix_t H = I;

        const auto op = [&] (solver_state_t& cstate, const std::size_t)
        {
                // descent direction
                cstate.d = -H * cstate.g;

                // line-search
                pstate = cstate;
                if (!lsearch(function, cstate))
                {
                        return false;
                }

                // update approximation of the Hessian
                const auto dx = cstate.x - pstate.x;
                const auto dg = cstate.g - pstate.g;
                const auto ro = scalar_t(1) / dx.dot(dg);

                H = (I - ro * dx * dg.transpose()) * H * (I - ro * dg * dx.transpose()) + ro * dx * dx.transpose();

                return true;
        };

        // assembly the solver
        return loop(function, x0, max_iterations, epsilon, logger, op);
}
