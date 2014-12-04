#pragma once

#include "batch_params.hpp"
#include "ls_wolfe.hpp"
#include "batch_cgd_steps.hpp"
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
                struct batch_cgd : public batch_params<tproblem>
                {
                        typedef batch_params<tproblem>          base_t;

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
                        batch_cgd(      tsize max_iterations,
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

                                tscalar ft;
                                tvector gt;

                                const tscalar alpha = tscalar(1e-4);
                                const tscalar beta = tscalar(0.1);

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
                                                cstate.d = -cstate.g + std::max(static_cast<tscalar>(0), beta) * pstate.d;
                                        }

                                        // update solution
                                        const tscalar t = ls_strong_wolfe(problem, cstate, base_t::m_wlog, ft, gt, alpha, beta);
                                        if (t < std::numeric_limits<tscalar>::epsilon())
                                        {
                                                base_t::elog(op_update.ls_failed_message());
                                                break;
                                        }
                                        pstate = cstate;
                                        cstate.update(problem, t, ft, gt);
                                }

                                return cstate;
                        }
                };

                // create various CGD algorithms
                template <typename tproblem>
                using batch_cgd_hs = batch_cgd<cgd_step_HS<typename tproblem::tstate>, tproblem>;

                template <typename tproblem>
                using batch_cgd_fr = batch_cgd<cgd_step_FR<typename tproblem::tstate>, tproblem>;

                template <typename tproblem>
                using batch_cgd_pr = batch_cgd<cgd_step_PR<typename tproblem::tstate>, tproblem>;

                template <typename tproblem>
                using batch_cgd_cd = batch_cgd<cgd_step_CD<typename tproblem::tstate>, tproblem>;

                template <typename tproblem>
                using batch_cgd_ls = batch_cgd<cgd_step_LS<typename tproblem::tstate>, tproblem>;

                template <typename tproblem>
                using batch_cgd_dy = batch_cgd<cgd_step_DY<typename tproblem::tstate>, tproblem>;

                template <typename tproblem>
                using batch_cgd_n = batch_cgd<cgd_step_N<typename tproblem::tstate>, tproblem>;
        }
}

