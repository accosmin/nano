#pragma once

#include "stoch_loop.hpp"

namespace nano
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
                        const auto op = [&] (const auto... params)
                        {
                                return this->operator()(param.tunable(), problem, x0, params...);
                        };

                        const auto beta1s = make_log10_space(std::log10(0.1), std::log10(0.9), 0.2);
                        const auto beta2s = make_log10_space(std::log10(0.9), std::log10(0.9999), 0.2);

                        const auto config = nano::tune(op, make_alpha0s(), make_epsilons(), beta1s, beta2s);
                        return operator()(param, problem, x0, config.param0(), config.param1(), config.param2(), config.param3());
                }

                ///
                /// \brief minimize starting from the initial guess x0
                ///
                tstate operator()(const param_t& param, const tproblem& problem, const tvector& x0,
                        const tscalar alpha0, const tscalar epsilon,
                        const tscalar beta1, const tscalar beta2) const
                {
                        assert(problem.size() == x0.size());

                        // first-order momentum of the gradient
                        momentum_vector_t<tvector> m(beta1, x0.size());

                        // second-order momentum of the gradient
                        momentum_vector_t<tvector> v(beta2, x0.size());

                        const auto op_iter = [&] (tstate& cstate, const std::size_t)
                        {
                                // learning rate
                                const tscalar alpha = alpha0;

                                // descent direction
                                m.update(cstate.g);
                                v.update(cstate.g.array().square());

                                cstate.d = -m.value().array() / (epsilon + v.value().array().sqrt());

                                // update solution
                                cstate.update(problem, alpha);
                        };

                        // OK, assembly the optimizer
                        return  stoch_loop(problem, param, tstate(problem, x0), op_iter,
                                {{"alpha0", alpha0}, {"epsilon", epsilon}, {"beta1", beta1}, {"beta2", beta2}});
                }
        };
}

