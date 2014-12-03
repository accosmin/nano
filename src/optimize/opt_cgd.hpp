#pragma once

#include "ls_wolfe.hpp"
#include "opt_cgd_steps.hpp"
#include <cassert>

namespace ncv
{
        namespace optimize
        {
                ///
                /// \brief conjugate gradient descent starting from the initial value (guess) x0
                ///
                template
                <
                        // CGD step update
                        typename tcgd_update,

                        // optimization problem
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
                        const tcgd_update& op_update,
                        const tvector& x0,
                        tsize max_iterations,           ///< maximum number of iterations
                        tscalar epsilon,                ///< convergence precision
                        const twlog& op_wlog = twlog(),
                        const telog& op_elog = telog(),
                        const tulog& op_ulog = tulog())
                {
                        assert(problem.size() == static_cast<tsize>(x0.size()));

                        tstate cstate(problem, x0);     // current state
                        tstate pstate = cstate;         // previous state

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
                                const tscalar t = ls_strong_wolfe(problem, cstate, op_wlog, ft, gt, tscalar(1e-4), tscalar(0.1));
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

                #define NCV_MAKE_CGD_OPTIMIZER(NAME, STEP) \
                template \
                < \
                        typename tproblem, \
                        \
                        typename tscalar = typename tproblem::tscalar, \
                        typename tsize = typename tproblem::tsize, \
                        typename tvector = typename tproblem::tvector, \
                        typename tstate = typename tproblem::tstate, \
                        \
                        typename twlog = typename tproblem::twlog, \
                        typename telog = typename tproblem::telog, \
                        typename tulog = typename tproblem::tulog \
                > \
                tstate cgd_##NAME( \
                        const tproblem& problem, \
                        const tvector& x0, \
                        tsize max_iterations, \
                        tscalar epsilon, \
                        const twlog& op_wlog = twlog(), \
                        const telog& op_elog = telog(), \
                        const tulog& op_ulog = tulog()) \
                { \
                        return  cgd(problem, cgd_step_##STEP<tstate>, \
                                x0, max_iterations, epsilon, op_wlog, op_elog, op_ulog); \
                }

                // instantiate various CGD algorithms
                NCV_MAKE_CGD_OPTIMIZER(hs, HS)
                NCV_MAKE_CGD_OPTIMIZER(fr, FR)
                NCV_MAKE_CGD_OPTIMIZER(pr, PR)
                NCV_MAKE_CGD_OPTIMIZER(cd, CD)
                NCV_MAKE_CGD_OPTIMIZER(ls, LS)
                NCV_MAKE_CGD_OPTIMIZER(dy, DY)
                NCV_MAKE_CGD_OPTIMIZER(n,  N)
        }
}

