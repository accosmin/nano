#pragma once

#include "stoch_loop.hpp"

namespace zob
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

                        const auto config = zob::tune(op, make_alpha0s(), make_epsilons());
                        return operator()(param, problem, x0, config.param0(), config.param1());
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

                        // OK, assembly the optimizer
                        return  stoch_loop(problem, param, tstate(problem, x0), op_iter,
                                {{"alpha0", alpha0}, {"epsilon", epsilon}, {"beta1", beta1}, {"beta2", beta2}});
                }
        };
}

