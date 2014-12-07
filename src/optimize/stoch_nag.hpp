#pragma once

#include "stoch_params.hpp"
#include <cassert>

namespace ncv
{
        namespace optimize
        {
                ///
                /// \brief stochastic Nesterov's accelerated gradient (descent)
                ///
                /// NB: Yu. Nesterov, "Introductory Lectures on Convex Optimization. A Basic Course"
                /// NB: http://calculus.subwiki.org/wiki/Nesterov%27s_accelerated_gradient_descent_with_constant_learning_rate_for_a_quadratic_function_of_one_variable
                /// NB: http://stronglyconvex.com/blog/accelerated-gradient-descent.html
                ///
                template
                <
                        typename tproblem               ///< optimization problem
                >
                struct stoch_nag : public stoch_params<tproblem>
                {
                        typedef stoch_params<tproblem>          base_t;

                        typedef typename base_t::tscalar        tscalar;
                        typedef typename base_t::tsize          tsize;
                        typedef typename base_t::tvector        tvector;
                        typedef typename base_t::tstate         tstate;
                        typedef typename base_t::tulog          tulog;

                        ///
                        /// \brief constructor
                        ///
                        stoch_nag(      tsize epochs,
                                        tsize epoch_size,
                                        tscalar alpha0,
                                        tscalar decay,
                                        const tulog& ulog = tulog())
                                :       base_t(epochs, epoch_size, alpha0, decay, ulog)
                        {
                        }

                        ///
                        /// \brief minimize starting from the initial guess x0
                        ///
                        tstate operator()(const tproblem& problem, const tvector& x0) const
                        {
                                assert(problem.size() == static_cast<tsize>(x0.size()));

                                tstate cstate;                  // current state

                                tvector py = x0;                // previous iteration
                                tvector cy = x0;                // current iteration

                                tvector px = x0;                // previous iteration
                                tvector cx = x0;                // current iteration

                                tvector g = x0;                 // gradient

                                for (tsize e = 0, k = 0; e < base_t::m_epochs; e ++)
                                {
                                        for (tsize i = 0; i < base_t::m_epoch_size; i ++)
                                        {
                                                // learning rate
                                                const tscalar alpha = base_t::alpha(k ++);

                                                // descent direction
                                                problem(py, g);
                                                const tscalar m = tscalar(k) / tscalar(k + 3);

                                                cx = py - alpha * g;
                                                cy = cx + m * (cx - px);

                                                // update solution
                                                cstate.x = cx;

                                                // next iteration
                                                px = cx;
                                                py = cy;
                                        }

                                        base_t::ulog(cstate);
                                }

                                return cstate;
                        }
                };
        }
}

