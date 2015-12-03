#pragma once

#include "params.hpp"
#include "best_state.hpp"
#include "ag_restarts.hpp"

namespace math
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

                        // restart method
                        trestart restart;

                        // current state
                        tstate cstate(problem, x0);

                        // best state
                        best_state_t<tstate> bstate(cstate);

                        // current & previous iterations
                        tvector cx = cstate.x;
                        tvector x1 = cstate.x;
                        tvector x2 = cstate.x;

                        tscalar fx = cstate.f;
                        tscalar f1 = cstate.f;

                        for (std::size_t e = 0, k = 1; e < m_param.m_epochs; ++ e)
                        {
                                for (std::size_t i = 0; i < m_param.m_epoch_size; ++ i, ++ k)
                                {
                                        // learning rate
                                        const tscalar alpha = m_param.alpha(0);

                                        // momentum
                                        const tscalar m = tscalar(k - 1) / tscalar(k + 2);

                                        // update solution
                                        cx = x1 + m * (x1 - x2);
                                        cstate.update(problem, cx);
                                        fx = cstate.f;

                                        k = restart(cstate.g, cx, fx, x1, f1, k);

                                        cx -= alpha * cstate.g;

                                        // next iteration
                                        x2 = x1;
                                        x1 = cx;
                                        f1 = fx;
                                }

                                cstate.update(problem, cx);
                                m_param.ulog(cstate);
                                bstate.update(cstate);
                        }

                        // OK
                        return bstate.get();
                }

                // attributes
                param_t         m_param;
        };

        // create various AG implementations
        template <typename tproblem>
        using stoch_ag_t =
        stoch_ag_base_t<tproblem, ag_no_restart_t<typename tproblem::tvector, typename tproblem::tscalar, std::size_t>>;

        template <typename tproblem>
        using stoch_agfr_t =
        stoch_ag_base_t<tproblem, ag_func_restart_t<typename tproblem::tvector, typename tproblem::tscalar, std::size_t>>;

        template <typename tproblem>
        using stoch_aggr_t =
        stoch_ag_base_t<tproblem, ag_grad_restart_t<typename tproblem::tvector, typename tproblem::tscalar, std::size_t>>;
}

