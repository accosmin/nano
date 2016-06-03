#pragma once

#include "loop.hpp"
#include "ls_init.hpp"
#include "cgd_steps.hpp"
#include "ls_strategy.hpp"

namespace nano
{
        ///
        /// \brief conjugate gradient descent
        ///
        template
        <
                typename tcgd_update                    ///< CGD step update
        >
        struct batch_cgd_t
        {
                ///
                /// \brief minimize starting from the initial guess x0
                ///
                state_t operator()(const batch_params_t& param, const problem_t& problem, const vector_t& x0) const
                {
                        assert(problem.size() == x0.size());

                        state_t istate(problem, x0);    // initial state
                        state_t pstate = istate;        // previous state

                        // line-search initial step length
                        ls_init_t ls_init(param.m_ls_initializer);

                        // line-search step
                        ls_strategy_t ls_step(param.m_ls_strategy, scalar_t(1e-4), scalar_t(0.1));

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

                                if (cstate.d.dot(cstate.g) > scalar_t(0))
                                {
                                        // reset to gradient descent, if not a descent direction!
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
        };

        // create various CGD algorithms
        using batch_cgd_hs_t = batch_cgd_t<cgd_step_HS>;
        using batch_cgd_fr_t = batch_cgd_t<cgd_step_FR>;
        using batch_cgd_prp_t = batch_cgd_t<cgd_step_PRP>;
        using batch_cgd_cd_t = batch_cgd_t<cgd_step_CD>;
        using batch_cgd_ls_t = batch_cgd_t<cgd_step_LS>;
        using batch_cgd_dy_t = batch_cgd_t<cgd_step_DY>;
        using batch_cgd_n_t = batch_cgd_t<cgd_step_N>;
        using batch_cgd_dycd_t = batch_cgd_t<cgd_step_DYCD>;
        using batch_cgd_dyhs_t = batch_cgd_t<cgd_step_DYHS>;
}

