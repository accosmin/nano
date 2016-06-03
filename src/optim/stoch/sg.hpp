#pragma once

#include "loop.hpp"
#include "lrate.hpp"

namespace nano
{
        ///
        /// \brief stochastic gradient (descent)
        ///     see "Minimizing Finite Sums with the Stochastic Average Gradient",
        ///     by Mark Schmidth, Nicolas Le Roux, Francis Bach
        ///
        struct stoch_sg_t
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
                        const auto config = nano::tune(op, param0, param1);
                        return operator()(param, problem, x0, config.param0(), config.param1());
                }

                ///
                /// \brief minimize starting from the initial guess x0
                ///
                state_t operator()(const stoch_params_t& param, const problem_t& problem, const vector_t& x0,
                        const scalar_t alpha0, const scalar_t decay) const
                {
                        assert(problem.size() == x0.size());

                        // learning rate schedule
                        lrate_t<scalar_t> lrate(alpha0, decay);

                        const auto op_iter = [&] (state_t& cstate)
                        {
                                // learning rate
                                const scalar_t alpha = lrate.get();

                                // descent direction
                                cstate.d = -cstate.g;

                                // update solution
                                cstate.update(problem, alpha);
                        };

                        // OK, assembly the optimizer
                        return  stoch_loop(problem, param, state_t(problem, x0), op_iter,
                                {{"alpha0", alpha0}, {"decay", decay}});
                }
        };
}

