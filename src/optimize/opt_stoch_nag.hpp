#pragma once

#include <cassert>

namespace ncv
{
        namespace optimize
        {
                ///
                /// \brief stochastic Nesterov's accelerated gradient (descent) starting from the initial value (guess) x0
                ///
                /// NB: Yu. Nesterov, "Introductory Lectures on Convex Optimization. A Basic Course"
                /// NB: http://calculus.subwiki.org/wiki/Nesterov%27s_accelerated_gradient_descent_with_constant_learning_rate_for_a_quadratic_function_of_one_variable
                ///
                template
                <
                        // optimization problem
                        typename tproblem,

                        // dependent types
                        typename tscalar = typename tproblem::tscalar,
                        typename tsize = typename tproblem::tsize,
                        typename tvector = typename tproblem::tvector,
                        typename tstate = typename tproblem::tstate,

                        typename tulog = typename tproblem::tulog
                >
                tstate stoch_nag(
                        const tproblem& problem,
                        const tvector& x0,
                        tsize epochs,                           ///< number of epochs
                        tsize iterations,                       ///< epoch size in number of iterations
                        tscalar alpha0,                         ///< initial learning rate
                        const tulog& op_ulog = tulog())         ///< called after each epoch with the current state
                {
                        assert(problem.size() == static_cast<tsize>(x0.size()));

                        tstate cstate;                  // current state

                        tvector py = x0;                // previous iteration
                        tvector cy = x0;                // current iteration

                        tvector px = x0;                // previous iteration
                        tvector cx = x0;                // current iteration

                        tvector g = x0;                 // gradient

                        for (tsize e = 0, k = 0; e < epochs; e ++)
                        {
                                for (tsize i = 0; i < iterations; i ++, k ++)
                                {
                                        // descent direction
                                        problem(py, g);
                                        const tscalar m = tscalar(k - 1) / tscalar(k + 2);

                                        cx = py - alpha0 * g;
                                        cy = px + m * (cx - px);

                                        // update solution
                                        cstate.x = cx;

                                        // next iteration
                                        px = cx;
                                        py = cy;
                                }

                                if (op_ulog)
                                {
                                        op_ulog(cstate);
                                }
                        }

                        return cstate;
                }
        }
}

