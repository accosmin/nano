#pragma once

#include "batch_params.hpp"
#include "linesearch_init.hpp"
#include "linesearch_strategy.hpp"
#include <cassert>

namespace ncv
{
        namespace optim
        {
                ///
                /// \brief gradient descent
                ///
                template
                <
                        typename tproblem                       ///< optimization problem
                >
                struct batch_gd_t : public batch_params_t<tproblem>
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
                        batch_gd_t(     tsize max_iterations,
                                        tscalar epsilon,
                                        ls_initializer lsinit,
                                        ls_strategy lsstrat,
                                        const twlog& wlog = twlog(),
                                        const telog& elog = telog(),
                                        const tulog& ulog = tulog())
                                :       base_t(max_iterations, epsilon, lsinit, lsstrat, wlog, elog, ulog)
                        {
                        }

                        ///
                        /// \brief minimize starting from the initial guess x0
                        ///
                        tstate operator()(const tproblem& problem, const tvector& x0) const
                        {
                                assert(problem.size() == static_cast<tsize>(x0.size()));

                                tstate cstate(problem, x0);             // current state

                                // line-search initial step length
                                linesearch_init_t<tstate> ls_init(base_t::m_ls_initializer);

                                // line-search step
                                linesearch_strategy_t<tproblem> ls_step(base_t::m_ls_strategy, 1e-4, 0.1);

                                // iterate until convergence
                                for (tsize i = 0; i < base_t::m_max_iterations && base_t::ulog(cstate); i ++)
                                {
                                        // check convergence
                                        if (cstate.converged(base_t::m_epsilon))
                                        {
                                                break;
                                        }

                                        // descent direction
                                        cstate.d = -cstate.g;

                                        // line-search
                                        const tscalar t0 = ls_init(cstate);
                                        if (!ls_step.update(problem, t0, cstate))
                                        {
                                                base_t::elog("line-search failed (GD)!");
                                                break;
                                        }
                                }

                                return cstate;
                        }
                };
        }
}

