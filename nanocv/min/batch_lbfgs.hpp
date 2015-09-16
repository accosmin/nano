#pragma once

#include "ls_init.hpp"
#include "ls_strategy.hpp"
#include "batch_params.hpp"
#include <deque>
#include <vector>

namespace ncv
{
        namespace min
        {
                ///
                /// \brief limited memory bfgs (l-bfgs)
                ///
                template
                <
                        typename tproblem                       ///< optimization problem
                >
                struct batch_lbfgs_t
                {
                        typedef batch_params_t<tproblem>        param_t;
                        typedef typename param_t::tscalar       tscalar;
                        typedef typename param_t::tsize         tsize;
                        typedef typename param_t::tvector       tvector;
                        typedef typename param_t::tstate        tstate;
                        typedef typename param_t::twlog         twlog;
                        typedef typename param_t::telog         telog;
                        typedef typename param_t::tulog         tulog;

                        ///
                        /// \brief constructor
                        ///
                        batch_lbfgs_t(  tsize max_iterations,
                                        tscalar epsilon,
                                        ls_initializer lsinit,
                                        ls_strategy lsstrat,
                                        tsize history_size,
                                        const twlog& wlog = twlog(),
                                        const telog& elog = telog(),
                                        const tulog& ulog = tulog())
                                :       m_param(max_iterations, epsilon, lsinit, lsstrat, wlog, elog, ulog),
                                        m_hsize(history_size)
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
                                linesearch_init_t<tstate> ls_init(m_param.m_ls_initializer);

                                // line-search step
                                linesearch_strategy_t<tproblem> ls_step(m_param.m_ls_strategy, 1e-4, 0.9);

                                // iterate until convergence
                                for (tsize i = 0; i < m_param.m_max_iterations && m_param.ulog(cstate); i ++)
                                {
                                        // check convergence
                                        if (cstate.converged(m_param.m_epsilon))
                                        {
                                                break;
                                        }

                                        // descent direction
                                        //      (see "Numerical optimization", Nocedal & Wright, 2nd edition, p.178)
                                        q = cstate.g;

                                        typename std::deque<tvector>::const_reverse_iterator itr_s = ss.rbegin();
                                        typename std::deque<tvector>::const_reverse_iterator itr_y = ys.rbegin();
                                        std::vector<tscalar> alphas;
                                        for (tsize j = 1; j <= m_hsize && i >= j; j ++)
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
                                        for (tsize j = 1; j <= m_hsize && i >= j; j ++)
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

                                        const tscalar t0 = ls_init(cstate);
                                        if (!ls_step.update(problem, t0, cstate))
                                        {
                                                m_param.elog("line-search failed (LBFGS)!");
                                                break;
                                        }

                                        ss.push_back(cstate.x - pstate.x);
                                        ys.push_back(cstate.g - pstate.g);
                                        if (ss.size() > m_hsize)
                                        {
                                                ss.pop_front();
                                                ys.pop_front();
                                        }
                                }

                                return cstate;
                        }

                        // attributes
                        param_t         m_param;
                        tsize           m_hsize;///< number of previous iterations to approximate Hessian's inverse
                };
        }
}

