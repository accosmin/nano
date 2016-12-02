#pragma once

#include "params.h"
#include "function.h"

namespace nano
{
        ///
        /// \brief batch optimization loop running until:
        ///     - convergence is achieved (critical point, possiblly a local/global minima) or
        ///     - the maximum number of iterations is reached or
        ///     - the user canceled the optimization (using the logging function) or
        ///     - the optimizer failed (e.g. line-search failed)
        ///
        template <typename toptimizer>
        auto batch_loop(
                const batch_params_t& params,
                const function_t& function,
                const vector_t& x0,
                const toptimizer& optimizer)
        {
                assert(function.size() == x0.size());

                // current state
                state_t cstate = make_state(function, x0);

                // for each iteration ...
                for (size_t i = 0; i < params.m_max_iterations; i ++)
                {
                        if (!optimizer(cstate, i) || !cstate)
                        {
                                cstate.m_status = opt_status::failed;
                                break;
                        }

                        // check convergence
                        if (cstate.converged(params.m_epsilon))
                        {
                                cstate.m_status = opt_status::converged;
                                params.ulog(cstate);
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

