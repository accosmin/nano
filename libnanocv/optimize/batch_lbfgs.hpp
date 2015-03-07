#pragma once

#include "batch_params.hpp"
#include "linesearch_init.hpp"
#include "linesearch_backtracking.hpp"
#include <deque>
#include <vector>
#include <cassert>

namespace ncv
{
        namespace optimize
        {
                ///
                /// \brief limited memory bfgs (l-bfgs)
                ///
                template
                <
                        typename tproblem                       ///< optimization problem
                >
                struct batch_lbfgs_t : public batch_params_t<tproblem>
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
                        batch_lbfgs_t(  tsize max_iterations,
                                        tscalar epsilon,
                                        ls_criterion lscrit,
                                        ls_initializer lsinit,
                                        tsize history_size,
                                        const twlog& wlog = twlog(),
                                        const telog& elog = telog(),
                                        const tulog& ulog = tulog())
                                :       base_t(max_iterations, epsilon, lscrit, lsinit, wlog, elog, ulog),
                                        m_history_size(history_size)
                        {
                        }

                        ///
                        /// \brief minimize starting from the initial guess x0
                        ///
                        tstate operator()(const tproblem& problem, const tvector& x0) const
                        {
                                assert(problem.size() == static_cast<tsize>(x0.size()));

                                std::deque<tvector> ss, ys;
                                tstate cstate(problem, x0);             // current state
                                tstate pstate = cstate;                 // previous state

                                tvector q, r;

                                // line-search initial step length
                                linesearch_init_t<tstate> ls_init(base_t::m_ls_initializer);

                                // line-search step
                                linesearch_backtracking_t<tproblem> ls_step(base_t::m_ls_criterion, 1e-4, 0.9);

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
                                        //      (LBFGS - Nocedal & Wright (numerical optimization 2nd) notations @ p.178)
                                        q = cstate.g;

                                        typename std::deque<tvector>::const_reverse_iterator itr_s = ss.rbegin();
                                        typename std::deque<tvector>::const_reverse_iterator itr_y = ys.rbegin();
                                        std::vector<tscalar> alphas;
                                        for (tsize j = 1; j <= m_history_size && i >= j; j ++)
                                        {
                                                const tvector& s = (*itr_s ++);
                                                const tvector& y = (*itr_y ++);

                                                const tscalar alpha = s.dot(q) / s.dot(y);
                                                q.noalias() -= alpha * y;
                                                alphas.push_back(alpha);
                                        }

                                        if (i == 0)
                                        {
                                                r = q;
                                        }
                                        else
                                        {
                                                const tvector& s = *ss.rbegin();
                                                const tvector& y = *ys.rbegin();
                                                r = s.dot(y) / y.dot(y) * q;
                                        }

                                        typename std::deque<tvector>::const_iterator it_s = ss.begin();
                                        typename std::deque<tvector>::const_iterator it_y = ys.begin();
                                        typename std::vector<tscalar>::const_reverse_iterator itr_alpha = alphas.rbegin();
                                        for (tsize j = 1; j <= m_history_size && i >= j; j ++)
                                        {
                                                const tvector& s = (*it_s ++);
                                                const tvector& y = (*it_y ++);

                                                const tscalar alpha = *(itr_alpha ++);
                                                const tscalar beta = y.dot(r) / s.dot(y);
                                                r.noalias() += s * (alpha - beta);
                                        }

                                        cstate.d = -r;

                                        // line-search
                                        pstate = cstate;

                                        const tscalar t0 = ls_init.update(cstate);
                                        if (!ls_step.update(problem, t0, cstate))
                                        {
                                                base_t::elog("line-search failed (LBFGS)!");
                                                break;
                                        }

                                        ss.push_back(cstate.x - pstate.x);
                                        ys.push_back(cstate.g - pstate.g);
                                        if (ss.size() > m_history_size)
                                        {
                                                ss.pop_front();
                                                ys.pop_front();
                                        }
                                }

                                return cstate;
                        }

                        tsize   m_history_size; ///< number of previous iterations to approximate the Hessian's inverse
                };
        }
}

