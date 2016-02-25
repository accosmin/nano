#pragma once

#include "stoch_loop.hpp"
#include "math/tune_fixed.hpp"

namespace math
{
        ///
        /// \brief stochastic Adam,
        ///     see "Adam: A method for stochastic optimization", by Diederik P. Kingma & Jimmy Lei Ba
        ///
        template
        <
                typename tproblem               ///< optimization problem
        >
        struct stoch_adam_t
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
                        const auto epsilons = { 1e-4, 1e-6, 1e-8 };

                        const auto op = [&] (const auto alpha0, const auto epsilon)
                        {
                                return this->operator()(param.tunable(), problem, x0, alpha0, epsilon);
                        };

                        const auto config = math::tune_fixed(op, alpha0s, epsilons);
                        const auto opt_alpha0 = std::get<1>(config);
                        const auto opt_epsilon = std::get<2>(config);

                        return operator()(param, problem, x0, opt_alpha0, opt_epsilon);
                }

                ///
                /// \brief minimize starting from the initial guess x0
                ///
                tstate operator()(const param_t& param, const tproblem& problem, const tvector& x0,
                        const tscalar alpha0, const tscalar epsilon) const
                {
                        assert(problem.size() == x0.size());

                        const tscalar beta1 = 0.900;
                        const tscalar beta2 = 0.999;

                        // first-order momentum of the gradient
                        momentum_vector_t<tvector> m(beta1, x0.size());

                        // second-order momentum of the gradient
                        momentum_vector_t<tvector> v(beta2, x0.size());

                        const auto op_iter = [&] (tstate& cstate, const std::size_t)
                        {
                                // learning rate
                                const auto alpha = alpha0;

                                // descent direction
                                m.update(cstate.g);
                                v.update(cstate.g.array().square());

                                cstate.d = -m.value().array() / (epsilon + v.value().array().sqrt());

                                // update solution
                                cstate.update(problem, alpha);
                        };

                        const auto op_epoch = [] (tstate&)
                        {
                        };

                        // OK, assembly the optimizer
                        return  stoch_loop(param, tstate(problem, x0), op_iter, op_epoch,
                                {{"alpha0", alpha0}, {"epsilon", epsilon}, {"beta1", beta1}, {"beta2", beta2}});
                }
        };
}

