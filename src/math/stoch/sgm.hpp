#pragma once

#include "lrate.hpp"
#include "math/tune.hpp"
#include "stoch_loop.hpp"
#include "math/momentum.hpp"

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
                        const auto op = [&] (const auto... params)
                        {
                                return this->operator()(param.tunable(), problem, x0, params...);
                        };

                        const auto alpha0s = math::make_log10_space(-4.0, +0.0, 0.20);
                        const auto decays = math::make_linear_space(0.10, 1.00, 0.05);
                        const auto momenta = math::make_linear_space(0.1, 0.99, 0.05);

                        const auto config = math::tune(op, alpha0s, decays, momenta);
                        return operator()(param, problem, x0, config.param0(), config.param1(), config.param2());
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
                        return  stoch_loop(param, tstate(problem, x0), op_iter, op_epoch,
                                {{"alpha0", alpha0}, {"decay", decay}, {"momentum", momentum}});
                }
        };
}

