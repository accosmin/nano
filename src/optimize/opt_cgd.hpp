#ifndef NANOCV_OPTIMIZE_OPTIMIZER_CGD_HPP
#define NANOCV_OPTIMIZE_OPTIMIZER_CGD_HPP

#include "ls_wolfe.hpp"
#include <cassert>

namespace ncv
{
        namespace optimize
        {
                /////////////////////////////////////////////////////////////////////////////////////////
                // conjugate gradient descent starting from the initial value (guess) x0.
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
                tstate cgd(
                        const tproblem& problem,
                        const tvector& x0,
                        tsize max_iterations,           // maximum number of iterations
                        tscalar epsilon,                // convergence precision
                        const twlog& op_wlog = twlog(),
                        const telog& op_elog = telog(),
                        const tulog& op_ulog = tulog())
                {
                        assert(problem.size() == static_cast<tsize>(x0.size()));

                        tstate cstate(problem, x0), pstate = cstate;

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
                                cstate.update(problem, t, ft, gt);
                        }

                        return cstate;
                }
        }
}

#endif // NANOCV_OPTIMIZE_OPTIMIZER_CGD_HPP
