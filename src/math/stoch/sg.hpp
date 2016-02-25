#pragma once

#include "lrate.hpp"
#include "stoch_loop.hpp"
#include "math/tune_fixed.hpp"

namespace math
{
        ///
        /// \brief stochastic gradient (descent)
        ///     see "Minimizing Finite Sums with the Stochastic Average Gradient",
        ///     by Mark Schmidth, Nicolas Le Roux, Francis Bach
        ///
        template
        <
                typename tproblem               ///< optimization problem
        >
        struct stoch_sg_t
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
                        const auto decays = { 0.10, 0.20, 0.50, 0.75, 1.00 };

                        const auto op = [&] (const auto alpha0, const auto decay)
                        {
                                return this->operator()(param.tunable(), problem, x0, alpha0, decay);
                        };

                        const auto config = math::tune_fixed(op, alpha0s, decays);
                        const auto opt_alpha0 = std::get<1>(config);
                        const auto opt_decay = std::get<2>(config);

                        return operator()(param, problem, x0, opt_alpha0, opt_decay);
                }

                ///
                /// \brief minimize starting from the initial guess x0
                ///
                tstate operator()(const param_t& param, const tproblem& problem, const tvector& x0,
                        const tscalar alpha0, const tscalar decay) const
                {
                        assert(problem.size() == x0.size());

                        // learning rate schedule
                        lrate_t<tscalar> lrate(alpha0, decay);

                        const auto op_iter = [&] (tstate& cstate, const std::size_t iter)
                        {
                                // learning rate
                                const tscalar alpha = lrate.get(iter);

                                // descent direction
                                cstate.d = -cstate.g;

                                // update solution
                                cstate.update(problem, alpha);
                        };

                        const auto op_epoch = [&] (tstate&)
                        {
                        };

                        // OK, assembly the optimizer
                        return  stoch_loop(param, tstate(problem, x0), op_iter, op_epoch,
                                {{"alpha0", alpha0}, {"decay", decay}});
                }
        };
}

