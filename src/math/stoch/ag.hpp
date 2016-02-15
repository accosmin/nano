#pragma once

#include "stoch_loop.hpp"

namespace math
{
        ///
        /// \brief restart methods for Nesterov's accelerated gradient
        ///     see "Adaptive Restart for Accelerated Gradient Schemes",
        ///     by Brendan O’Donoghue & Emmanuel Candes, 2013
        ///
        enum class ag_restart
        {
                none,
                function,
                gradient
        };

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
                ag_restart trestart             ///< restart method
        >
        struct stoch_ag_base_t
        {
                using param_t = stoch_params_t<tproblem>;
                using tstate = typename param_t::tstate;
                using tscalar = typename param_t::tscalar;
                using tvector = typename param_t::tvector;
                using topulog = typename param_t::topulog;

                ///
                /// \brief constructor
                ///
                explicit stoch_ag_base_t(const param_t& param) : m_param(param)
                {
                }

                ///
                /// \brief minimize starting from the initial guess x0
                ///
                tstate operator()(const tproblem& problem, const tvector& x0) const
                {
                        assert(problem.size() == x0.size());

                        // initial state
                        tstate istate(problem, x0);

                        // current & previous iterations
                        tvector cx = istate.x;
                        tvector px = istate.x;
                        tvector cy = istate.x;
                        tvector py = istate.x;

                        tscalar cfx = istate.f;
                        tscalar pfx = istate.f;

                        const auto op_iter = [&] (tstate& cstate, std::size_t& k)
                        {
                                // learning rate
                                const tscalar alpha = m_param.alpha(0);

                                // momentum
                                const tscalar m = tscalar(k - 1) / tscalar(k + 2);

                                // update solution
                                cstate.update(problem, py);
                                cx = py - alpha * cstate.g;
                                cy = px + m * (cx - px);

                                switch (trestart)
                                {
                                case ag_restart::none:
                                        break;

                                case ag_restart::function:
                                        if ((cfx = problem(cx)) > pfx)
                                        {
                                                k = 0;
                                        }
                                        break;

                                case ag_restart::gradient:
                                        if (cstate.g.dot(cx - px) > tscalar(0))
                                        {
                                                k = 0;
                                        }
                                        break;
                                }

                                // next iteration
                                px = cx;
                                py = cy;
                                pfx = cfx;
                        };

                        const auto op_epoch = [&] (tstate& cstate)
                        {
                                cstate.update(problem, cx);
                        };

                        // OK, assembly the optimizer
                        return stoch_loop(m_param, istate, op_iter, op_epoch);
                }

                // attributes
                param_t         m_param;
        };

        // create various AG implementations
        template <typename tproblem>
        using stoch_ag_t = stoch_ag_base_t<tproblem, ag_restart::none>;

        template <typename tproblem>
        using stoch_agfr_t = stoch_ag_base_t<tproblem, ag_restart::function>;

        template <typename tproblem>
        using stoch_aggr_t = stoch_ag_base_t<tproblem, ag_restart::gradient>;
}

