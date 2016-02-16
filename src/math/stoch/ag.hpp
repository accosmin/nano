#pragma once

#include "stoch_loop.hpp"
#include "math/tune_fixed.hpp"

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
        ///     see "Adaptive Restart for Accelerated Gradient Schemes",
        ///     by Brendan O’Donoghue & Emmanuel Candes, 2013
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

                ///
                /// \brief minimize starting from the initial guess x0
                ///
                tstate operator()(const param_t& param, const tproblem& problem, const tvector& x0) const
                {
                        const auto alpha0s = { 1e-4, 1e-3, 1e-2, 1e-1, 1e+0 };
                        const auto qs = { 1e-6, 1e-4, 1e-2, 1e-1, 1e+0 };

                        const auto op = [&] (const auto alpha0, const auto q)
                        {
                                return this->operator()(param.tunable(), problem, x0, alpha0, q);
                        };

                        const auto config = math::tune_fixed(op, alpha0s, qs);
                        const auto opt_alpha0 = std::get<1>(config);
                        const auto opt_q = std::get<2>(config);

                        return operator()(param, problem, x0, opt_alpha0, opt_q);
                }

                ///
                /// \brief minimize starting from the initial guess x0
                ///
                tstate operator()(const param_t& param, const tproblem& problem, const tvector& x0,
                        const tscalar alpha0, const tscalar q) const
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

                        tscalar ptheta = 1;
                        tscalar ctheta = 1;

                        const auto get_theta = [] (const auto ptheta, const auto q)
                        {
                                const auto a = tscalar(1);
                                const auto b = ptheta * ptheta - q;
                                const auto c = - ptheta * ptheta;

                                return (-b + std::sqrt(b * b - 4 * a * c)) / (2 * a);
                        };

                        const auto get_beta = [] (const auto ptheta, const auto ctheta)
                        {
                                return ptheta * (1 - ptheta) / (ptheta * ptheta + ctheta);
                        };

                        const auto op_iter = [&] (tstate& cstate, const std::size_t)
                        {
                                // learning rate
                                const tscalar alpha = alpha0;

                                // momentum
                                ctheta = get_theta(ptheta, q);
                                const tscalar beta = get_beta(ptheta, ctheta);

                                // update solution
                                cstate.update(problem, py);
                                cx = py - alpha * cstate.g;
                                cy = px + beta * (cx - px);

                                switch (trestart)
                                {
                                case ag_restart::none:
                                        break;

                                case ag_restart::function:
                                        if ((cfx = problem(cx)) > pfx)
                                        {
                                                ptheta = 1;
                                        }
                                        break;

                                case ag_restart::gradient:
                                        if (cstate.g.dot(cx - px) > tscalar(0))
                                        {
                                                ptheta = 1;
                                        }
                                        break;
                                }

                                // next iteration
                                px = cx;
                                py = cy;
                                pfx = cfx;
                                ptheta = ctheta;
                        };

                        const auto op_epoch = [&] (tstate& cstate)
                        {
                                cstate.update(problem, cx);
                        };

                        // OK, assembly the optimizer
                        return stoch_loop(param, istate, op_iter, op_epoch);
                }
        };

        // create various AG implementations
        template <typename tproblem>
        using stoch_ag_t = stoch_ag_base_t<tproblem, ag_restart::none>;

        template <typename tproblem>
        using stoch_agfr_t = stoch_ag_base_t<tproblem, ag_restart::function>;

        template <typename tproblem>
        using stoch_aggr_t = stoch_ag_base_t<tproblem, ag_restart::gradient>;
}

