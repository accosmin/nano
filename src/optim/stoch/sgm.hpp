#pragma once

#include "loop.hpp"
#include "lrate.hpp"
#include "math/momentum.hpp"

namespace nano
{
        ///
        /// \brief stochastic gradient (descent) with momentum
        ///
        struct stoch_sgm_t
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
                        const auto param2 = make_momenta();
                        const auto config = nano::tune(op, param0, param1, param2);
                        return operator()(param, problem, x0, config.param0(), config.param1(), config.param2());
                }

                ///
                /// \brief minimize starting from the initial guess x0
                ///
                state_t operator()(const stoch_params_t& param, const problem_t& problem, const vector_t& x0,
                        const scalar_t alpha0, const scalar_t decay, const scalar_t momentum) const
                {
                        assert(problem.size() == x0.size());

                        // learning rate schedule
                        lrate_t<scalar_t> lrate(alpha0, decay);

                        // first-order momentum of the update
                        momentum_vector_t<vector_t> davg(momentum, x0.size());

                        const auto op_iter = [&] (state_t& cstate)
                        {
                                // learning rate
                                const scalar_t alpha = lrate.get();

                                // descent direction
                                davg.update(-alpha * cstate.g);
                                cstate.d = davg.value();

                                // update solution
                                cstate.update(problem, 1);
                        };

                        // OK, assembly the optimizer
                        return  stoch_loop(problem, param, state_t(problem, x0), op_iter,
                                {{"alpha0", alpha0}, {"decay", decay}, {"momentum", momentum}});
                }
        };
}

