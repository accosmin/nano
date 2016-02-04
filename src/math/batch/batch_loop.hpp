#pragma once

#include "params.hpp"

namespace math
{
        ///
        /// \brief batch optimization loop
        ///
        template
        <
                typename tparams,       ///< optimization parameters
                typename tstate,        ///< initial state
                typename top_iter       ///< operator to call for each optimization iteration
        >
        tstate batch_loop(const tparams& params, const tstate& istate, const top_iter& opi)
        {
                // current state
                tstate cstate = istate;

                // for each iteration ...
                for (std::size_t i = 0; i < params.m_max_iterations && params.ulog(cstate); i ++)
                {
                        // check convergence
                        if (cstate.converged(params.m_epsilon))
                        {
                                break;
                        }

                        if (!opi(cstate, i))
                        {
                                break;
                        }

                        // log the current state & check the stopping criteria
                        if (!params.ulog(cstate))
                        {
                                break;
                        }
                }

                // OK
                return cstate;
        }
}

