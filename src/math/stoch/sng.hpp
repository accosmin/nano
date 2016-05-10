#pragma once

#include "stoch_loop.hpp"

namespace nano
{
        ///
        /// \brief stochastic normalized gradient (descent)
        ///     see "Beyond Convexity: Stochastic Quasi-Convex Optimization",
        ///     by Elan Hazan, Kfir Y. Levi, Shai Shalev-Shwartz
        ///
        template
        <
                typename tproblem               ///< optimization problem
        >
        struct stoch_sng_t
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

                        const auto config = nano::tune(op, make_alpha0s());
                        return operator()(param, problem, x0, config.param0());
                }

                ///
                /// \brief minimize starting from the initial guess x0
                ///
                tstate operator()(const param_t& param, const tproblem& problem, const tvector& x0,
                        const tscalar alpha0) const
                {
                        assert(problem.size() == x0.size());

                        const auto op_iter = [&] (tstate& cstate)
                        {
                                // learning rate
                                const tscalar alpha = alpha0;

                                // descent direction
                                const tscalar norm = 1 / cstate.g.template lpNorm<2>();
                                cstate.d = -cstate.g * norm;

                                // update solution
                                cstate.update(problem, alpha);
                        };

                        // OK, assembly the optimizer
                        return  stoch_loop(problem, param, tstate(problem, x0), op_iter,
                                {{"alpha0", alpha0}});
                }
        };
}

