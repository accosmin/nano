#pragma once

#include "params.h"

namespace nano
{
        ///
        /// \brief batch optimization loop running until:
        ///     - convergence is achieved (critical point, possiblly a local/global minima) or
        ///     - the maximum number of iterations is reached or
        ///     - the user canceled the optimization (using the logging function) or
        ///     - the optimizer failed (e.g. line-search failed)
        ///
        template
        <
                typename toptimizer     ///< optimization algorithm
        >
        auto batch_loop(
                const batch_params_t& params,
                const state_t& istate,
                const toptimizer& optimizer)
        {
                // current state
                auto cstate = istate;

                // for each iteration ...
                for (size_t i = 0; i < params.m_max_iterations; i ++)
                {
                        // check convergence
                        if (cstate.converged(params.m_epsilon))
                        {
                                cstate.m_status = opt_status::converged;
                                break;
                        }

                        if (!optimizer(cstate, i))
                        {
                                cstate.m_status = opt_status::failed;
                                break;
                        }

                        // log the current state & check the stopping criteria
                        if (!params.ulog(cstate))
                        {
                                cstate.m_status = opt_status::stopped;
                                break;
                        }
                }

                // OK
                return cstate;
        }
}

