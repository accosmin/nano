#pragma once

#include "params.hpp"

namespace zob
{
        ///
        /// \brief stochastic optimization loop
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

