#include "solver_quasi.h"
#include <deque>

using namespace nano;

template <typename tquasi_update>
tuner_t solver_quasi_base_t<tquasi_update>::tuner() const
{
        tuner_t tuner;
        tuner.add_enum<lsearch_t::initializer>("init");
        tuner.add_enum<lsearch_t::strategy>("strat");
        return tuner;
}

template <typename tquasi_update>
void solver_quasi_base_t<tquasi_update>::from_json(const json_t& json)
{
        nano::from_json(json,
                "init", m_init, "strat", m_strat,
                "c1", m_c1, "c2", m_c2);
}

template <typename tquasi_update>
void solver_quasi_base_t<tquasi_update>::to_json(json_t& json) const
{
        nano::to_json(json,
                "init", m_init, "inits", join(enum_values<lsearch_t::initializer>()),
                "strat", m_strat, "strats", join(enum_values<lsearch_t::strategy>()),
                "c1", m_c1, "c2", m_c2);
}

template <typename tquasi_update>
solver_state_t solver_quasi_base_t<tquasi_update>::minimize(const size_t max_iterations, const scalar_t epsilon,
        const function_t& f, const vector_t& x0, const logger_t& logger) const
{
        lsearch_t lsearch(m_init, m_strat, m_c1, m_c2);

        // previous state
        solver_state_t pstate(f.size());

        // current approximation of the Hessian
        matrix_t H = matrix_t::Identity(f.size(), f.size());

        const auto op = [&] (const function_t& function, solver_state_t& cstate, const size_t)
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
                H = tquasi_update::get(H, pstate, cstate);
                return true;
        };

        // assembly the solver
        return loop(f, x0, max_iterations, epsilon, logger, op);
}

template class nano::solver_quasi_base_t<quasi_step_DFP>;
template class nano::solver_quasi_base_t<quasi_step_SR1>;
template class nano::solver_quasi_base_t<quasi_step_BFGS>;
template class nano::solver_quasi_base_t<quasi_step_broyden>;
