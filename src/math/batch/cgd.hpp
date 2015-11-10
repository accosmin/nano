#pragma once

#include "params.hpp"
#include "ls_init.hpp"
#include "cgd_steps.hpp"
#include "ls_strategy.hpp"

namespace math
{
        ///
        /// \brief conjugate gradient descent
        ///
        template
        <
                typename tcgd_update,                   ///< CGD step update
                typename tproblem                       ///< optimization problem
        >
        struct batch_cgd_t
        {
                using param_t = batch_params_t<tproblem>;
                using tstate = typename param_t::tstate;
                using tscalar = typename param_t::tscalar;
                using tvector = typename param_t::tvector;
                using topulog = typename param_t::topulog;

                ///
                /// \brief constructor
                ///
                batch_cgd_t(    std::size_t max_iterations,
                                tscalar epsilon,
                                ls_initializer lsinit,
                                ls_strategy lsstrat,
                                const topulog& ulog = topulog())
                        :       m_param(max_iterations, epsilon, lsinit, lsstrat, ulog)
                {
                }

                ///
                /// \brief minimize starting from the initial guess x0
                ///
                tstate operator()(const tproblem& problem, const tvector& x0) const
                {
                        assert(problem.size() == x0.size());

                        tstate cstate(problem, x0);     // current state
                        tstate pstate = cstate;         // previous state

                        // line-search initial step length
                        ls_init_t<tstate> ls_init(m_param.m_ls_initializer);

                        // line-search step
                        ls_strategy_t<tproblem> ls_step(m_param.m_ls_strategy, 1e-4, 0.1);

                        const tcgd_update op_update;

                        // iterate until convergence
                        for (std::size_t i = 0; i < m_param.m_max_iterations && m_param.ulog(cstate); i ++)
                        {
                                // check convergence
                                if (cstate.converged(m_param.m_epsilon))
                                {
                                        break;
                                }

                                // descent direction
                                if (i == 0)
                                {
                                        cstate.d = -cstate.g;
                                }
                                else
                                {
                                        const tscalar beta = op_update(pstate, cstate);
                                        cstate.d = -cstate.g + beta * pstate.d;
                                }

                                if (cstate.d.dot(cstate.g) > tscalar(0))
                                {
                                        // reset to gradient descent, if not a descent direction!
                                        cstate.d = -cstate.g;
                                }

                                // line-search
                                pstate = cstate;

                                const tscalar t0 = ls_init(cstate);
                                if (!ls_step(problem, t0, cstate))
                                {
                                        break;
                                }
                        }

                        // OK
                        return cstate;
                }

                // attributes
                param_t         m_param;
        };

        // create various CGD algorithms
        template <typename tproblem>
        using batch_cgd_hs_t = batch_cgd_t<cgd_step_HS<typename tproblem::tstate>, tproblem>;

        template <typename tproblem>
        using batch_cgd_fr_t = batch_cgd_t<cgd_step_FR<typename tproblem::tstate>, tproblem>;

        template <typename tproblem>
        using batch_cgd_prp_t = batch_cgd_t<cgd_step_PRP<typename tproblem::tstate>, tproblem>;

        template <typename tproblem>
        using batch_cgd_cd_t = batch_cgd_t<cgd_step_CD<typename tproblem::tstate>, tproblem>;

        template <typename tproblem>
        using batch_cgd_ls_t = batch_cgd_t<cgd_step_LS<typename tproblem::tstate>, tproblem>;

        template <typename tproblem>
        using batch_cgd_dy_t = batch_cgd_t<cgd_step_DY<typename tproblem::tstate>, tproblem>;

        template <typename tproblem>
        using batch_cgd_n_t = batch_cgd_t<cgd_step_N<typename tproblem::tstate>, tproblem>;

        template <typename tproblem>
        using batch_cgd_dycd_t = batch_cgd_t<cgd_step_DYCD<typename tproblem::tstate>, tproblem>;

        template <typename tproblem>
        using batch_cgd_dyhs_t = batch_cgd_t<cgd_step_DYHS<typename tproblem::tstate>, tproblem>;
}

