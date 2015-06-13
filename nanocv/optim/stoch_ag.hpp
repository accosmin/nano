#pragma once

#include "stoch_params.hpp"
#include "stoch_ag_restarts.hpp"

namespace ncv
{
        namespace optim
        {
                ///
                /// \brief stochastic Nesterov's accelerated gradient (descent)
                ///     see "A method of solving a convex programming problem with convergence rate O(1/k^2)",
                ///     by Yu. Nesterov, 1983
                ///
                ///     see "Gradient methods for minimizing composite objective function",
                ///     by Yu. Nesterov, 2007
                ///
                ///     see "Adaptive Restart for Accelerated Gradient Schemes",
                ///     by Brendan O’Donoghue & Emmanuel Candes, 2013
                ///
                ///     see "A Differential Equation for Modeling Nesterov’s Accelerated Gradient Method:
                ///     Theory and Insights",
                ///     by Weijie Su, Stephen Boyd & Emmanuel J. Candes, 2015
                ///
                ///     see http://calculus.subwiki.org/wiki/Nesterov%27s_accelerated_gradient_descent_with_constant_learning_rate_for_a_quadratic_function_of_one_variable
                ///     see http://stronglyconvex.com/blog/accelerated-gradient-descent.html
                ///
                template
                <
                        typename tproblem,              ///< optimization problem
                        typename trestart               ///< restart method
                >
                struct stoch_ag_base_t : public stoch_params_t<tproblem>
                {
                        typedef stoch_params_t<tproblem>        base_t;

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
                        stoch_ag_base_t(tsize epochs,
                                        tsize epoch_size,
                                        tscalar alpha0,
                                        tscalar decay,
                                        const twlog& wlog = twlog(),
                                        const telog& elog = telog(),
                                        const tulog& ulog = tulog())
                                :       base_t(epochs, epoch_size, alpha0, decay, wlog, elog, ulog)
                        {
                        }

                        ///
                        /// \brief minimize starting from the initial guess x0
                        ///
                        tstate operator()(const tproblem& problem, const tvector& x0) const
                        {
                                assert(problem.size() == static_cast<tsize>(x0.size()));

                                // restart method
                                trestart restart;

                                // current state
                                tstate cstate(problem, x0);

                                // previous & current iteration
                                tvector y = x0;

                                // previous / current iteration
                                tvector px = x0;
                                tvector cx = x0;

                                for (tsize e = 0, k = 1; e < base_t::m_epochs; e ++)
                                {
                                        for (tsize i = 0; i < base_t::m_epoch_size; i ++, k ++)
                                        {
                                                // learning rate
                                                const tscalar alpha = base_t::m_alpha0;

                                                // descent direction
                                                problem(y, cstate.g);
                                                const tscalar m = tscalar(k - 1) / tscalar(k + 2);

                                                cx = y - alpha * cstate.g;
                                                y = cx + m * (cx - px);

                                                // update solution
                                                cstate.x = cx;

                                                // next iteration
                                                restart(cstate.g, cx, px, k);
                                                px = cx;
                                        }

                                        base_t::ulog(cstate);
                                }

                                return cstate;
                        }
                };

                // create various AG implementations
                template <typename tproblem>
                using stoch_ag_t =
                stoch_ag_base_t<tproblem, ag_no_restart_t<typename tproblem::tvector, typename tproblem::tsize>>;

                template <typename tproblem>
                using stoch_aggr_t =
                stoch_ag_base_t<tproblem, ag_grad_restart_t<typename tproblem::tvector, typename tproblem::tsize>>;
        }
}

