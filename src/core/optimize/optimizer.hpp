#ifndef NANOCV_OPTIMIZE_OPTIMIZER_HPP
#define NANOCV_OPTIMIZE_OPTIMIZER_HPP

#include "core/optimize/linesearch.hpp"
#include "core/optimize/problem.hpp"
#include <deque>

namespace ncv
{
        namespace optimize
        {
                /////////////////////////////////////////////////////////////////////////////////////////
                // check and force a descent direction
                /////////////////////////////////////////////////////////////////////////////////////////

                template
                <
                        typename tstate,
                        typename twlog,

                        typename tscalar = typename tstate::tscalar
                >
                tscalar descent(tstate& st, const twlog& wlog)
                {
                        tscalar dg = st.d.dot(st.g);
                        if (dg > std::numeric_limits<tscalar>::min())
                        {
                                if (wlog)
                                {
                                        wlog("not a descent direction!");
                                }
                                st.d = -st.g;
                                dg = st.d.dot(st.g);
                        }

                        return dg;
                }

                /////////////////////////////////////////////////////////////////////////////////////////
                // gradient descent starting from the initial value (guess) x0
                /////////////////////////////////////////////////////////////////////////////////////////

                template
                <
                        typename tproblem,

                        // dependent types
                        typename tscalar = typename tproblem::tscalar,
                        typename tsize = typename tproblem::tsize,
                        typename tvector = typename tproblem::tvector,
                        typename tresult = typename tproblem::tresult,
                        typename tstate = typename tproblem::tstate,

                        typename twlog = typename tproblem::twlog,
                        typename telog = typename tproblem::telog,
                        typename tulog = typename tproblem::tulog
                >
                tresult gd(
                        const tproblem& problem,
                        const tvector& x0,
                        tsize max_iterations,
                        tscalar epsilon,
                        const twlog& op_wlog = twlog(),
                        const telog& op_elog = telog(),
                        const tulog& op_ulog = tulog())
                {
                        assert(problem.size() == static_cast<tsize>(x0.size()));

                        tresult result(problem.size());
                        tstate cstate(problem, x0);

                        // iterate until convergence
                        for (tsize i = 0; i < max_iterations; i ++)
                        {
                                result.update(problem, cstate);
                                if (op_ulog)
                                {
                                        op_ulog(result);
                                }

                                // check convergence
                                if (cstate.converged(epsilon))
                                {
                                        break;
                                }

                                // descent direction
                                cstate.d = -cstate.g;

                                // update solution
                                const tscalar t = ls_armijo(problem, cstate, op_wlog);
                                if (t < std::numeric_limits<tscalar>::epsilon())
                                {
                                        if (op_elog)
                                        {
                                                op_elog("line-search failed for GD!");
                                        }
                                        break;
                                }
                                cstate.update(problem, t);
                        }

                        return result;
                }

                /////////////////////////////////////////////////////////////////////////////////////////
                // conjugate gradient descent starting from the initial value (guess) x0
                /////////////////////////////////////////////////////////////////////////////////////////

                template
                <
                        typename tproblem,

                        // dependent types
                        typename tscalar = typename tproblem::tscalar,
                        typename tsize = typename tproblem::tsize,
                        typename tvector = typename tproblem::tvector,
                        typename tresult = typename tproblem::tresult,
                        typename tstate = typename tproblem::tstate,

                        typename twlog = typename tproblem::twlog,
                        typename telog = typename tproblem::telog,
                        typename tulog = typename tproblem::tulog
                >
                tresult cgd(
                        const tproblem& problem,
                        const tvector& x0,
                        tsize max_iterations,           // maximum number of iterations
                        tscalar epsilon,                // convergence precision
                        const twlog& op_wlog = twlog(),
                        const telog& op_elog = telog(),
                        const tulog& op_ulog = tulog())
                {
                        assert(problem.size() == static_cast<tsize>(x0.size()));

                        tresult result(problem.size());
                        tstate cstate(problem, x0), pstate = cstate;

                        tscalar ft;
                        tvector gt;

                        // iterate until convergence
                        for (tsize i = 0; i < max_iterations; i ++)
                        {
                                result.update(problem, cstate);
                                if (op_ulog)
                                {
                                        op_ulog(result);
                                }

                                // check convergence
                                if (cstate.converged(epsilon))
                                {
                                        break;
                                }

                                // descent direction (Polak–Ribière updates)
                                if (i == 0)
                                {
                                        cstate.d = -cstate.g;
                                }
                                else
                                {
                                        const tscalar beta = cstate.g.dot(cstate.g - pstate.g) /
                                                              pstate.g.dot(pstate.g);
                                        cstate.d = -cstate.g + std::max(static_cast<tscalar>(0), beta) * pstate.d;
                                }

                                // update solution
                                const tscalar t = ls_strong_wolfe(problem, cstate, op_wlog, ft, gt, 1e-4, 0.1);
                                if (t < std::numeric_limits<tscalar>::epsilon())
                                {
                                        if (op_elog)
                                        {
                                                op_elog("line-search failed for CGD!");
                                        }
                                        break;
                                }
                                pstate = cstate;
                                cstate.update(t, ft, gt);
                        }

                        return result;
                }

                /////////////////////////////////////////////////////////////////////////////////////////
                // limited memory bfgs (l-bfgs) starting from the initial value (guess) x0
                /////////////////////////////////////////////////////////////////////////////////////////

                template
                <
                        typename tproblem,

                        // dependent types
                        typename tscalar = typename tproblem::tscalar,
                        typename tsize = typename tproblem::tsize,
                        typename tvector = typename tproblem::tvector,
                        typename tresult = typename tproblem::tresult,
                        typename tstate = typename tproblem::tstate,

                        typename twlog = typename tproblem::twlog,
                        typename telog = typename tproblem::telog,
                        typename tulog = typename tproblem::tulog
                >
                tresult lbfgs(
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

                        tresult result(problem.size());
                        std::deque<tvector> ss, ys;
                        tstate cstate(problem, x0), pstate = cstate;

                        tvector q, r;
                        tscalar ft;
                        tvector gt;

                        // iterate until convergence
                        for (tsize i = 0; i < max_iterations; i ++)
                        {
                                result.update(problem, cstate);
                                if (op_ulog)
                                {
                                        op_ulog(result);
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
                                cstate.update(t, ft, gt);

                                ss.push_back(cstate.x - pstate.x);
                                ys.push_back(cstate.g - pstate.g);
                                if (ss.size() > hist_size)
                                {
                                        ss.pop_front();
                                        ys.pop_front();
                                }
                        }

                        return result;
                }
        }
}

#endif // NANOCV_OPTIMIZE_OPTIMIZER_HPP
