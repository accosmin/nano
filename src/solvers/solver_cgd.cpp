#include "solver_cgd.h"

using namespace nano;

template <typename tcgd_update>
tuner_t solver_cgd_base_t<tcgd_update>::tuner() const
{
        tuner_t tuner;
        tuner.add_finite("c1", 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1);
        tuner.add_finite("c2", 0.2, 0.5, 0.9);
        tuner.add_finite("orthotest", 1e-4, 1e-3, 1e-2, 1e-1);
        tuner.add_enum<lsearch_t::initializer>("init");
        tuner.add_enum<lsearch_t::strategy>("strat");
        return tuner;
}

template <typename tcgd_update>
void solver_cgd_base_t<tcgd_update>::from_json(const json_t& json)
{
        nano::from_json(json,
                "init", m_init, "strat", m_strat, "c1", m_c1, "c2", m_c2, "orthotest", m_orthotest);
}

template <typename tcgd_update>
void solver_cgd_base_t<tcgd_update>::to_json(json_t& json) const
{
        nano::to_json(json,
                "init", m_init, "inits", join(enum_values<lsearch_t::initializer>()),
                "strat", m_strat, "strats", join(enum_values<lsearch_t::strategy>()),
                "c1", m_c1, "c2", m_c2, "orthotest", m_orthotest);
}

template <typename tcgd_update>
solver_state_t solver_cgd_base_t<tcgd_update>::minimize(const size_t max_iterations, const scalar_t epsilon,
        const function_t& function, const vector_t& x0, const logger_t& logger) const
{
        lsearch_t lsearch(m_init, m_strat, m_c1, m_c2);

        // previous state
        solver_state_t pstate(function.size());

        // CGD direction strategy
        const tcgd_update op_update{};

        const auto op = [&] (solver_state_t& cstate, const std::size_t i)
        {
                // descent direction
                if (i == 0)
                {
                        cstate.d = -cstate.g;
                }
                else
                {
                        const scalar_t beta = op_update(pstate, cstate);
                        cstate.d = -cstate.g + beta * pstate.d;
                }

                // restart:
                //  - if not a descent direction
                //  - or two consecutive gradients far from being orthogonal
                //      (see "Numerical optimization", Nocedal & Wright, 2nd edition, p.124-125)
                if (cstate.d.dot(cstate.g) > scalar_t(0))
                {
                        cstate.d = -cstate.g;
                }
                else if (std::fabs(cstate.g.dot(pstate.g)) >= m_orthotest * cstate.g.dot(cstate.g))
                {
                        cstate.d = -cstate.g;
                }

                // line-search
                pstate = cstate;
                return lsearch(function, cstate);
        };

        // assembly the solver
        return loop(function, x0, max_iterations, epsilon, logger, op);
}

template class nano::solver_cgd_base_t<cgd_step_HS>;
template class nano::solver_cgd_base_t<cgd_step_FR>;
template class nano::solver_cgd_base_t<cgd_step_PRP>;
template class nano::solver_cgd_base_t<cgd_step_CD>;
template class nano::solver_cgd_base_t<cgd_step_LS>;
template class nano::solver_cgd_base_t<cgd_step_DY>;
template class nano::solver_cgd_base_t<cgd_step_N>;
template class nano::solver_cgd_base_t<cgd_step_DYCD>;
template class nano::solver_cgd_base_t<cgd_step_DYHS>;
