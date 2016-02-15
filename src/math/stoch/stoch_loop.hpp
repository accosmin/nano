#pragma once

#include "params.hpp"

namespace math
{
        ///
        /// \brief stochastic optimization loop
        ///
        template
        <
                typename tparam,        ///< optimization parameters
                typename tstate,        ///< initial state
                typename top_iter,      ///< operator to call for each optimization iteration
                typename top_epoch      ///< operator to call after each epoch
        >
        tstate stoch_loop(const tparam& params, const tstate& istate, const top_iter& opi, const top_epoch& ope)
        {
                // current state
                tstate cstate = istate;

                // best state
                tstate bstate = istate;

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
                        bstate.update(cstate);

                        // log the current state & check the stopping criteria
                        if (!params.ulog(cstate))
                        {
                                break;
                        }
                }

                // OK
                return bstate;
        }
}

