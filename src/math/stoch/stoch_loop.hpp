#pragma once

#include "params.hpp"
#include "math/tune.hpp"

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
                return zob::make_finite_space(0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90);
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
                typename top_iter,      ///< operator to call for each optimization iteration
                typename top_epoch      ///< operator to call after each epoch
        >
        auto stoch_loop(
                const stoch_params_t<tproblem>& params,
                const typename stoch_params_t<tproblem>::tstate& istate,
                const top_iter& opi,
                const top_epoch& ope,
                const typename stoch_params_t<tproblem>::tconfig& config)
        {
                // current state
                auto cstate = istate;

                // best state
                auto bstate = istate;

                // for each epoch ...
                for (std::size_t e = 0, k = 1; e < params.m_epochs; ++ e)
                {
                        // for each iteration ...
                        for (std::size_t i = 0; i < params.m_epoch_size; ++ i, ++ k)
                        {
                                opi(cstate, k);
                        }

                        // update the current & best states
                        ope(cstate);
                        cstate.f = params.tlog(cstate, config);
                        bstate.update(cstate);

                        // log the current state & check the stopping criteria
                        if (!params.ulog(cstate, config))
                        {
                                break;
                        }
                }

                // OK
                return bstate;
        }
}

