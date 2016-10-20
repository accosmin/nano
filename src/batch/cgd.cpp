#include "cgd.h"
#include "loop.h"
#include "ls_init.h"
#include "ls_strategy.h"
#include "text/to_params.h"
#include "text/from_params.h"

namespace nano
{
        template <typename tcgd_update>
        batch_cgd_t<tcgd_update>::batch_cgd_t(const string_t& configuration) :
                batch_optimizer_t(concat_params(configuration, "ls_init=init-quadratic,ls_strat=interpolation,c1=1e-4,c2=0.1"))
        {
        }

        template <typename tcgd_update>
        rbatch_optimizer_t batch_cgd_t<tcgd_update>::clone(const string_t& configuration) const
        {
                return std::make_unique<batch_cgd_t<tcgd_update>>(configuration);
        }

        template <typename tcgd_update>
        rbatch_optimizer_t batch_cgd_t<tcgd_update>::clone() const
        {
                return std::make_unique<batch_cgd_t<tcgd_update>>(*this);
        }

        template <typename tcgd_update>
        state_t batch_cgd_t<tcgd_update>::minimize(const batch_params_t& param, const problem_t& problem, const vector_t& x0) const
        {
                assert(problem.size() == x0.size());

                state_t istate(problem, x0);    // initial state
                state_t pstate = istate;        // previous state

                // line-search initial step length
                ls_init_t ls_init(from_params<ls_initializer>(config(), "ls_init"));

                // line-search step
                const auto c1 = from_params<scalar_t>(config(), "c1");
                const auto c2 = from_params<scalar_t>(config(), "c2");
                ls_strategy_t ls_step(from_params<ls_strategy>(config(), "ls_strat"), c1, c2);

                // CGD direction strategy
                const tcgd_update op_update{};

                const auto op = [&] (state_t& cstate, const std::size_t i)
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
                        return ls_step(problem, t0, cstate);
                };

                // OK, assembly the optimizer
                return batch_loop(param, istate, op);
        }

        template struct batch_cgd_t<cgd_step_HS>;
        template struct batch_cgd_t<cgd_step_FR>;
        template struct batch_cgd_t<cgd_step_PRP>;
        template struct batch_cgd_t<cgd_step_CD>;
        template struct batch_cgd_t<cgd_step_LS>;
        template struct batch_cgd_t<cgd_step_DY>;
        template struct batch_cgd_t<cgd_step_N>;
        template struct batch_cgd_t<cgd_step_DYCD>;
        template struct batch_cgd_t<cgd_step_DYHS>;
}

