#ifndef NANOCV_OPTIMIZE_OPTIMIZER_GD_HPP
#define NANOCV_OPTIMIZE_OPTIMIZER_GD_HPP

#include "ls_armijo.hpp"
#include <cassert>

namespace ncv
{
        namespace optimize
        {
                /////////////////////////////////////////////////////////////////////////////////////////
                // gradient descent starting from the initial value (guess) x0.
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
                tstate gd(
                        const tproblem& problem,
                        const tvector& x0,
                        tsize max_iterations,
                        tscalar epsilon,
                        const twlog& op_wlog = twlog(),
                        const telog& op_elog = telog(),
                        const tulog& op_ulog = tulog())
                {
                        assert(problem.size() == static_cast<tsize>(x0.size()));

                        tstate cstate(problem, x0);

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

                        return cstate;
                }
        }
}

#endif // NANOCV_OPTIMIZE_OPTIMIZER_GD_HPP
