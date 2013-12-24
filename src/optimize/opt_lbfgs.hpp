#ifndef NANOCV_OPTIMIZE_OPTIMIZER_LBFGS_HPP
#define NANOCV_OPTIMIZE_OPTIMIZER_LBFGS_HPP

#include "ls_wolfe.hpp"
#include <deque>
#include <cassert>

namespace ncv
{
        namespace optimize
        {
                /////////////////////////////////////////////////////////////////////////////////////////
                // limited memory bfgs (l-bfgs) starting from the initial value (guess) x0.
                /////////////////////////////////////////////////////////////////////////////////////////

                template
                <
                        typename tproblem,

                        // dependent types
                        typename tscalar = typename tproblem::tscalar,
                        typename tsize = typename tproblem::tsize,
                        typename tvector = typename tproblem::tvector,
                        typename tstate = typename tproblem::tstate,

                        typename twlog = typename tproblem::twlog,
                        typename telog = typename tproblem::telog,
                        typename tulog = typename tproblem::tulog
                >
                tstate lbfgs(
                        const tproblem& problem,
                        const tvector& x0,
                        tsize max_iterations,           // maximum number of iterations
                        tscalar epsilon,                // convergence precision
                        const twlog& op_wlog = twlog(),
                        const telog& op_elog = telog(),
                        const tulog& op_ulog = tulog())
                {
                        assert(problem.size() == static_cast<tsize>(x0.size()));

                        const tsize hist_size = tsize(6);

                        std::deque<tvector> ss, ys;
                        tstate cstate(problem, x0), pstate = cstate;

                        tvector q, r;
                        tscalar ft;
                        tvector gt;

                        // iterate until convergence
                        for (tsize i = 0; i < max_iterations; i ++)
                        {
                                if (op_ulog)
                                {
                                        op_ulog(cstate);
                                }

                                // check convergence
                                if (cstate.converged(epsilon))
                                {
                                        break;
                                }

                                // descent direction
                                //      (LBFGS - Nocedal & Wright (numerical optimization 2nd) notations @ p.178)
                                q = cstate.g;

                                typename std::deque<tvector>::const_reverse_iterator itr_s = ss.rbegin();
                                typename std::deque<tvector>::const_reverse_iterator itr_y = ys.rbegin();
                                std::vector<tscalar> alphas;
                                for (tsize j = 1; j <= hist_size && i >= j; j ++)
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
                                for (tsize j = 1; j <= hist_size && i >= j; j ++)
                                {
                                        const tvector& s = (*it_s ++);
                                        const tvector& y = (*it_y ++);

                                        const tscalar alpha = *(itr_alpha ++);
                                        const tscalar beta = y.dot(r) / s.dot(y);
                                        r.noalias() += s * (alpha - beta);
                                }

                                cstate.d = -r;

                                // update solution
                                const tscalar t = ls_strong_wolfe(problem, cstate, op_wlog, ft, gt, 1e-4, 0.9);
                                if (t < std::numeric_limits<tscalar>::epsilon())
                                {
                                        if (op_elog)
                                        {
                                                op_elog("line-search failed for LBFGS!");
                                        }
                                        break;
                                }
                                pstate = cstate;
                                cstate.update(problem, t, ft, gt);

                                ss.push_back(cstate.x - pstate.x);
                                ys.push_back(cstate.g - pstate.g);
                                if (ss.size() > hist_size)
                                {
                                        ss.pop_front();
                                        ys.pop_front();
                                }
                        }

                        return cstate;
                }
        }
}

#endif // NANOCV_OPTIMIZE_OPTIMIZER_LBFGS_HPP
