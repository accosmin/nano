#pragma once

#include "math/tune.hpp"
#include "stoch_loop.hpp"
#include "math/momentum.hpp"

namespace math
{
        ///
        /// \brief stochastic AdaDelta,
        ///     see "ADADELTA: An Adaptive Learning Rate Method", by Matthew D. Zeiler
        ///
        template
        <
                typename tproblem               ///< optimization problem
        >
        struct stoch_adadelta_t
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
                        const auto momenta = math::make_finite_space({ 0.90, 0.95, 0.99 });
                        const auto epsilons = math::make_finite_space({ 1e-4, 1e-6, 1e-8 });

                        const auto op = [&] (const auto momentum, const auto epsilon)
                        {
                                return this->operator()(param.tunable(), problem, x0, momentum, epsilon);
                        };

                        const auto config = math::tune(op, momenta, epsilons);
                        const auto opt_momentum = std::get<1>(config);
                        const auto opt_epsilon = std::get<2>(config);

                        return operator()(param, problem, x0, opt_momentum, opt_epsilon);
                }

                ///
                /// \brief minimize starting from the initial guess x0
                ///
                tstate operator()(const param_t& param, const tproblem& problem, const tvector& x0,
                        const tscalar momentum, const tscalar epsilon) const
                {
                        assert(problem.size() == x0.size());

                        // second-order momentum of the gradient
                        momentum_vector_t<tvector> gavg(momentum, x0.size());

                        // second-order momentum of the step updates
                        momentum_vector_t<tvector> davg(momentum, x0.size());

                        const auto op_iter = [&] (tstate& cstate, const std::size_t)
                        {
                                // learning rate
                                const tscalar alpha = 1;

                                // descent direction
                                gavg.update(cstate.g.array().square());

                                cstate.d = -cstate.g.array() *
                                           (epsilon + davg.value().array()).sqrt() /
                                           (epsilon + gavg.value().array()).sqrt();

                                davg.update(cstate.d.array().square());

                                // update solution
                                cstate.update(problem, alpha);
                        };

                        const auto op_epoch = [] (tstate&)
                        {
                        };

                        // OK, assembly the optimizer
                        return  stoch_loop(param, tstate(problem, x0), op_iter, op_epoch,
                                {{"momentum", momentum}, {"epsilon", epsilon}});
                }
        };
}

