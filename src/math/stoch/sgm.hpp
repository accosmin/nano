#pragma once

#include "lrate.hpp"
#include "stoch_loop.hpp"
#include "math/momentum.hpp"

namespace nano
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

                        const auto config = nano::tune(op, make_alpha0s(), make_decays(), make_momenta());
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

                        // first-order momentum of the update
                        momentum_vector_t<tvector> davg(momentum, x0.size());

                        const auto op_iter = [&] (tstate& cstate, const std::size_t iter)
                        {
                                // learning rate
                                const tscalar alpha = lrate.get(iter);

                                // descent direction
                                davg.update(-alpha * cstate.g);
                                cstate.d = davg.value();

                                // update solution
                                cstate.update(problem, 1);
                        };

                        // OK, assembly the optimizer
                        return  stoch_loop(problem, param, tstate(problem, x0), op_iter,
                                {{"alpha0", alpha0}, {"decay", decay}, {"momentum", momentum}});
                }
        };
}

