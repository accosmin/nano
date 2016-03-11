#pragma once

#include "stoch_loop.hpp"
#include "math/average.hpp"

namespace nano
{
        ///
        /// \brief stochastic AdaGrad
        ///     see "Adaptive subgradient methods for online learning and stochastic optimization"
        ///     by J. C. Duchi, E. Hazan, and Y. Singer
        ///
        ///     see http://xcorr.net/2014/01/23/adagrad-eliminating-learning-rates-in-stochastic-gradient-descent/
        ///
        template
        <
                typename tproblem               ///< optimization problem
        >
        struct stoch_adagrad_t
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

                        const auto config = nano::tune(op, make_alpha0s(), make_epsilons());
                        return operator()(param, problem, x0, config.param0(), config.param1());
                }

                ///
                /// \brief minimize starting from the initial guess x0
                ///
                tstate operator()(const param_t& param, const tproblem& problem, const tvector& x0,
                        const tscalar alpha0, const tscalar epsilon) const
                {
                        assert(problem.size() == x0.size());

                        // second-order gradient momentum
                        average_vector_t<tvector> gavg(x0.size());

                        const auto op_iter = [&] (tstate& cstate, const std::size_t)
                        {
                                // learning rate
                                const tscalar alpha = alpha0;

                                // descent direction
                                gavg.update(cstate.g.array().square());

                                cstate.d = -cstate.g.array() /
                                           (epsilon + gavg.value().array()).sqrt();

                                // update solution
                                cstate.update(problem, alpha);
                        };

                        // OK, assembly the optimizer
                        return  stoch_loop(problem, param, tstate(problem, x0), op_iter,
                                {{"alpha0", alpha0}, {"epsilon", epsilon}});
                }
        };
}

