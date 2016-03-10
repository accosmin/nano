#pragma once

#include "params.hpp"
#include "math/tune.hpp"
#include "math/momentum.hpp"

namespace zob
{
        ///
        /// \brief hyper-parameter tuning for stochastic optimizers.
        ///
        inline auto make_alpha0s()
        {
                return zob::make_finite_space(1e-4, 1e-3, 1e-2, 1e-1, 1e+0);
        }

        inline auto make_decays()
        {
                return zob::make_finite_space(0.10, 0.25, 0.50, 0.75, 1.00);
        }

        inline auto make_momenta()
        {
                return zob::make_finite_space(0.10, 0.25, 0.50, 0.90, 0.95);
        }

        inline auto make_epsilons()
        {
                return zob::make_finite_space(1e-4, 1e-6, 1e-8);
        }

        ///
        /// \brief stochastic optimization loop.
        ///
        template
        <
                typename tproblem,      ///< optimization problem
                typename toperator      ///< operator to call for each optimization iteration
        >
        auto stoch_loop(
                const tproblem& problem,
                const stoch_params_t<tproblem>& params,
                const typename stoch_params_t<tproblem>::tstate& istate,
                const toperator& op,
                const typename stoch_params_t<tproblem>::tconfig& config)
        {
                // current state
                auto cstate = istate;

                // average state
                auto astate = istate;

                const typename tproblem::tscalar momentum = 0.90;
                momentum_vector_t<typename tproblem::tvector> xavg(momentum, istate.x.size());

                // best state
                auto bstate = istate;

                // for each epoch ...
                for (std::size_t e = 0, k = 1; e < params.m_epochs; ++ e)
                {
                        // for each iteration ...
                        for (std::size_t i = 0; i < params.m_epoch_size; ++ i, ++ k)
                        {
                                op(cstate, k);
                                xavg.update(cstate.x);
                        }

                        // log the current state & check the stopping criteria
                        astate.update(problem, xavg.value());
                        if (!params.ulog(astate, config))
                        {
                                break;
                        }

                        // update the best state
                        bstate.update(astate);
                }

                // OK
                return bstate;
        }
}

