#pragma once

#include "lrate.hpp"
#include "stoch_loop.hpp"

namespace nano
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
                        const auto op = [&] (const auto... params)
                        {
                                return this->operator()(param.tunable(), problem, x0, params...);
                        };

                        const auto param0 = make_alpha0s<tscalar>();
                        const auto param1 = make_decays<tscalar>();
                        const auto param2 = make_finite_space(tscalar(0.05), tscalar(0.10), tscalar(0.15), tscalar(0.20));
                        const auto config = nano::tune(op, param0, param1, param2);
                        return operator()(param, problem, x0, config.param0(), config.param1(), config.param2());
                }

                ///
                /// \brief minimize starting from the initial guess x0
                ///
                tstate operator()(const param_t& param, const tproblem& problem, const tvector& x0,
                        const tscalar alpha0, const tscalar decay, const tscalar q) const
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

                        // learning rate schedule
                        lrate_t<tscalar> lrate(alpha0, decay);

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

                        const auto op_iter = [&] (tstate& cstate)
                        {
                                // learning rate
                                const tscalar alpha = lrate.get();

                                // momentum
                                ctheta = get_theta(ptheta, q);
                                const tscalar beta = get_beta(ptheta, ctheta);

                                // update solution
                                cstate.update(problem, py);
                                cx = py - alpha * cstate.g;
                                cy = cx + beta * (cx - px);
                                cstate.x = cx; // NB: to propagate the current parameters!

                                switch (trestart)
                                {
                                case ag_restart::none:
                                        break;

                                case ag_restart::function:
                                        if ((cfx = problem(cx)) > pfx)
                                        {
                                                ctheta = 1;
                                        }
                                        break;

                                case ag_restart::gradient:
                                        if (cstate.g.dot(cx - px) > tscalar(0))
                                        {
                                                ctheta = 1;
                                        }
                                        break;
                                }

                                // next iteration
                                px = cx;
                                py = cy;
                                pfx = cfx;
                                ptheta = ctheta;
                        };

                        // OK, assembly the optimizer
                        return  stoch_loop(problem, param, istate, op_iter,
                                {{"alpha0", alpha0}, {"decay", decay}, {"q", q}});
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

