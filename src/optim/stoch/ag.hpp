#pragma once

#include "loop.hpp"
#include "lrate.hpp"

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
                ag_restart trestart             ///< restart method
        >
        struct stoch_ag_base_t
        {
                ///
                /// \brief minimize starting from the initial guess x0
                ///
                state_t operator()(const stoch_params_t& param, const problem_t& problem, const vector_t& x0) const
                {
                        const auto op = [&] (const auto... params)
                        {
                                return this->operator()(param.tunable(), problem, x0, params...);
                        };

                        const auto param0 = make_alpha0s();
                        const auto param1 = make_decays();
                        const auto param2 = make_finite_space(scalar_t(0.05), scalar_t(0.10), scalar_t(0.15), scalar_t(0.20));
                        const auto config = nano::tune(op, param0, param1, param2);
                        return operator()(param, problem, x0, config.param0(), config.param1(), config.param2());
                }

                ///
                /// \brief minimize starting from the initial guess x0
                ///
                state_t operator()(const stoch_params_t& param, const problem_t& problem, const vector_t& x0,
                        const scalar_t alpha0, const scalar_t decay, const scalar_t q) const
                {
                        assert(problem.size() == x0.size());

                        // initial state
                        state_t istate(problem, x0);

                        // current & previous iterations
                        vector_t cx = istate.x;
                        vector_t px = istate.x;
                        vector_t cy = istate.x;
                        vector_t py = istate.x;

                        scalar_t cfx = istate.f;
                        scalar_t pfx = istate.f;

                        scalar_t ptheta = 1;
                        scalar_t ctheta = 1;

                        // learning rate schedule
                        lrate_t<scalar_t> lrate(alpha0, decay);

                        const auto get_theta = [] (const auto ptheta, const auto q)
                        {
                                const auto a = scalar_t(1);
                                const auto b = ptheta * ptheta - q;
                                const auto c = - ptheta * ptheta;

                                return (-b + std::sqrt(b * b - 4 * a * c)) / (2 * a);
                        };

                        const auto get_beta = [] (const auto ptheta, const auto ctheta)
                        {
                                return ptheta * (1 - ptheta) / (ptheta * ptheta + ctheta);
                        };

                        const auto op_iter = [&] (state_t& cstate)
                        {
                                // learning rate
                                const scalar_t alpha = lrate.get();

                                // momentum
                                ctheta = get_theta(ptheta, q);
                                const scalar_t beta = get_beta(ptheta, ctheta);

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
                                        if (cstate.g.dot(cx - px) > scalar_t(0))
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
        using stoch_ag_t = stoch_ag_base_t<ag_restart::none>;
        using stoch_agfr_t = stoch_ag_base_t<ag_restart::function>;
        using stoch_aggr_t = stoch_ag_base_t<ag_restart::gradient>;
}

