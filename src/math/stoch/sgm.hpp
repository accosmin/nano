#pragma once

#include "lrate.hpp"
#include "stoch_loop.hpp"
#include "math/momentum.hpp"
#include "math/tune_fixed.hpp"

namespace math
{
        ///
        /// \brief stochastic gradient (descent) with momentum
        ///
        template
        <
                typename tproblem               ///< optimization problem
        >
        struct stoch_sgm_t
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
                        const auto momenta = { 0.90, 0.95, 0.99 };

                        const auto op = [&] (const auto alpha0, const auto decay, const auto momentum)
                        {
                                return this->operator()(param.tunable(), problem, x0, alpha0, decay, momentum);
                        };

                        const auto config = math::tune_fixed(op, alpha0s, decays, momenta);
                        const auto opt_alpha0 = std::get<1>(config);
                        const auto opt_decay = std::get<2>(config);
                        const auto opt_momentum = std::get<3>(config);

                        return operator()(param, problem, x0, opt_alpha0, opt_decay, opt_momentum);
                }

                ///
                /// \brief minimize starting from the initial guess x0
                ///
                tstate operator()(const param_t& param, const tproblem& problem, const tvector& x0,
                        const tscalar alpha0, const tscalar decay, const tscalar momentum) const
                {
                        assert(problem.size() == x0.size());

                        // learning rate schedule
                        lrate_t<tscalar> lrate(alpha0, decay);

                        // first-order momentum of the gradient
                        momentum_vector_t<tvector> gavg(momentum, x0.size());

                        const auto op_iter = [&] (tstate& cstate, const std::size_t iter)
                        {
                                // learning rate
                                const tscalar alpha = lrate.get(iter);

                                // descent direction
                                gavg.update(cstate.g.array());
                                cstate.d = -gavg.value();

                                // update solution
                                cstate.update(problem, alpha);
                        };

                        const auto op_epoch = [&] (tstate&)
                        {
                        };

                        // OK, assembly the optimizer
                        return stoch_loop(param, tstate(problem, x0), op_iter, op_epoch);
                }
        };
}

