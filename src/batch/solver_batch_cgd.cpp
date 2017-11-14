#include "ls_init.h"
#include "ls_strategy.h"
#include "solver_batch_cgd.h"

using namespace nano;

template <typename tcgd_update>
json_reader_t& batch_cgd_t<tcgd_update>::config(json_reader_t& reader)
{
        return reader.object("ls_init", m_ls_init, "ls_strat", m_ls_strat, "c1", m_c1, "c2", m_c2);
}

template <typename tcgd_update>
json_writer_t& batch_cgd_t<tcgd_update>::config(json_writer_t& writer) const
{
        return writer.object(
                "ls_init", m_ls_init, "ls_inits", join(enum_values<ls_initializer>()),
                "ls_strat", m_ls_strat, "ls_strats", join(enum_values<ls_strategy>()),
                "c1", m_c1, "c2", m_c2);
}

template <typename tcgd_update>
solver_state_t nano::batch_cgd_t<tcgd_update>::minimize(const batch_params_t& param,
        const function_t& function, const vector_t& x0) const
{
        // previous state
        solver_state_t pstate(function.size());

        // line-search initial step length
        ls_init_t ls_init(m_ls_init);

        // line-search step
        ls_strategy_t ls_step(m_ls_strat, m_c1, m_c2);

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
                else if (std::fabs(cstate.g.dot(pstate.g)) >= param.m_cgd_orthotest * cstate.g.dot(cstate.g))
                {
                        cstate.d = -cstate.g;
                }

                // line-search
                pstate = cstate;

                const scalar_t t0 = ls_init(cstate);
                return ls_step(function, t0, cstate);
        };

        // assembly the solver
        return loop(param, function, x0, op);
}

template class nano::batch_cgd_t<cgd_step_HS>;
template class nano::batch_cgd_t<cgd_step_FR>;
template class nano::batch_cgd_t<cgd_step_PRP>;
template class nano::batch_cgd_t<cgd_step_CD>;
template class nano::batch_cgd_t<cgd_step_LS>;
template class nano::batch_cgd_t<cgd_step_DY>;
template class nano::batch_cgd_t<cgd_step_N>;
template class nano::batch_cgd_t<cgd_step_DYCD>;
template class nano::batch_cgd_t<cgd_step_DYHS>;
