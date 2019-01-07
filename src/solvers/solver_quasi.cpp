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
                "init", to_string(m_init) + join(enum_values<lsearch_t::initializer>()),
                "strat", to_string(m_strat) + join(enum_values<lsearch_t::strategy>()),
                "c1", m_c1, "c2", m_c2);
}

template <typename tquasi_update>
solver_state_t solver_quasi_base_t<tquasi_update>::minimize(const size_t max_iterations, const scalar_t epsilon,
        const solver_function_t& function, const vector_t& x0, const logger_t& logger) const
{
        lsearch_t lsearch(m_init, m_strat, m_c1, m_c2);

        // previous state
        solver_state_t pstate(function.size());

        // current approximation of the Hessian
        matrix_t H = matrix_t::Identity(function.size(), function.size());

        auto cstate = solver_state_t{function, x0};
        for (size_t i = 0; i < max_iterations; ++ i, ++ cstate.m_iterations)
        {
                // descent direction
                cstate.d = -H * cstate.g;

                // restart:
                //  - if not a descent direction
                if (!(cstate.d.dot(cstate.g) < 0))
                {
                        cstate.d = -cstate.g;
                        H.setIdentity();
                }

                // line-search
                pstate = cstate;
                const auto iter_ok = lsearch(function, cstate);
                if (solver_t::done(logger, function, cstate, epsilon, iter_ok))
                {
                        break;
                }

                // update approximation of the Hessian
                H = tquasi_update::get(H, pstate, cstate);
        }

        return cstate;
}

template class nano::solver_quasi_base_t<quasi_step_DFP>;
template class nano::solver_quasi_base_t<quasi_step_SR1>;
template class nano::solver_quasi_base_t<quasi_step_BFGS>;
template class nano::solver_quasi_base_t<quasi_step_broyden>;
