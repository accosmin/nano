#pragma once

#include "stoch_loop.hpp"
#include "math/momentum.hpp"

namespace nano
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
                        const auto op = [&] (const auto... params)
                        {
                                return this->operator()(param.tunable(), problem, x0, params...);
                        };

                        const auto param0 = make_momenta<tscalar>();
                        const auto param1 = make_epsilons<tscalar>();
                        const auto config = nano::tune(op, param0, param1);
                        return operator()(param, problem, x0, config.param0(), config.param1());
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

                        const auto op_iter = [&] (tstate& cstate)
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

                        // OK, assembly the optimizer
                        return  stoch_loop(problem, param, tstate(problem, x0), op_iter,
                                {{"momentum", momentum}, {"epsilon", epsilon}});
                }
        };
}

