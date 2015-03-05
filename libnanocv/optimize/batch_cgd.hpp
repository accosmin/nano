#pragma once

#include "batch_params.hpp"
#include "batch_cgd_steps.hpp"
#include "linesearch_init_interp.hpp"
#include "linesearch_wolfe.hpp"
#include <cassert>

namespace ncv
{
        namespace optimize
        {
                ///
                /// \brief conjugate gradient descent
                ///
                template
                <
                        typename tcgd_update,                   ///< CGD step update
                        typename tproblem                       ///< optimization problem
                >
                struct batch_cgd_t : public batch_params_t<tproblem>
                {
                        typedef batch_params_t<tproblem>        base_t;

                        typedef typename base_t::tscalar        tscalar;
                        typedef typename base_t::tsize          tsize;
                        typedef typename base_t::tvector        tvector;
                        typedef typename base_t::tstate         tstate;
                        typedef typename base_t::twlog          twlog;
                        typedef typename base_t::telog          telog;
                        typedef typename base_t::tulog          tulog;

                        ///
                        /// \brief constructor
                        ///
                        batch_cgd_t(    tsize max_iterations,
                                        tscalar epsilon,
                                        const twlog& wlog = twlog(),
                                        const telog& elog = telog(),
                                        const tulog& ulog = tulog())
                                :       base_t(max_iterations, epsilon, wlog, elog, ulog)
                        {
                        }

                        ///
                        /// \brief minimize starting from the initial guess x0
                        ///
                        tstate operator()(const tproblem& problem, const tvector& x0) const
                        {
                                assert(problem.size() == static_cast<tsize>(x0.size()));

                                tstate cstate(problem, x0);     // current state
                                tstate pstate = cstate;         // previous state

                                // line-search initial step length
                                linesearch_init_interpolation_t<tstate> ls_init;

                                // line-search step
                                linesearch_wolfe_t<tproblem> ls_step(1e-4, 0.1);

                                const tcgd_update op_update;

                                // iterate until convergence
                                for (tsize i = 0; i < base_t::m_max_iterations; i ++)
                                {
                                        base_t::ulog(cstate);

                                        // check convergence
                                        if (cstate.converged(base_t::m_epsilon))
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
                                                cstate.d = -cstate.g;
                                                base_t::wlog("not a descent direction (CGD)!");
                                        }

                                        // line-search
                                        pstate = cstate;

                                        const tscalar t0 = ls_init.update(cstate);
                                        if (!ls_step.update(problem, t0, cstate))
                                        {
                                                base_t::elog("line-search failed (CGD)!");
                                                break;
                                        }
                                }

                                return cstate;
                        }
                };

                // create various CGD algorithms
                template <typename tproblem>
                using batch_cgd_hs_t = batch_cgd_t<cgd_step_HS<typename tproblem::tstate>, tproblem>;

                template <typename tproblem>
                using batch_cgd_fr_t = batch_cgd_t<cgd_step_FR<typename tproblem::tstate>, tproblem>;

                template <typename tproblem>
                using batch_cgd_pr_t = batch_cgd_t<cgd_step_PR<typename tproblem::tstate>, tproblem>;

                template <typename tproblem>
                using batch_cgd_cd_t = batch_cgd_t<cgd_step_CD<typename tproblem::tstate>, tproblem>;

                template <typename tproblem>
                using batch_cgd_ls_t = batch_cgd_t<cgd_step_LS<typename tproblem::tstate>, tproblem>;

                template <typename tproblem>
                using batch_cgd_dy_t = batch_cgd_t<cgd_step_DY<typename tproblem::tstate>, tproblem>;

                template <typename tproblem>
                using batch_cgd_n_t = batch_cgd_t<cgd_step_N<typename tproblem::tstate>, tproblem>;
        }
}

