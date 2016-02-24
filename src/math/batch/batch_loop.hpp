#pragma once

#include "params.hpp"

namespace math
{
        ///
        /// \brief batch optimization loop
        ///
        template
        <
                typename tproblem,      ///< optimization problem
                typename top_iter       ///< operator to call for each optimization iteration
        >
        auto batch_loop(
                const batch_params_t<tproblem>& params,
                const typename tproblem::tstate& istate,
                const top_iter& opi)
        {
                // current state
                auto cstate = istate;

                // for each iteration ...
                for (std::size_t i = 0; i < params.m_max_iterations; i ++)
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

