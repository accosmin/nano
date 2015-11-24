#pragma once

#include "params.hpp"
#include "best_state.hpp"
#include "math/average.hpp"

namespace math
{
        ///
        /// \brief stochastic iterative average gradient (descent)
        ///     see "Minimizing Finite Sums with the Stochastic Average Gradient",
        ///     by Mark Schmidth, Nicolas Le Roux, Francis Bach
        ///
        template
        <
                typename tproblem               ///< optimization problem
        >
        struct stoch_sia_t
        {
                using param_t = stoch_params_t<tproblem>;
                using tstate = typename param_t::tstate;
                using tscalar = typename param_t::tscalar;
                using tvector = typename param_t::tvector;
                using topulog = typename param_t::topulog;

                ///
                /// \brief constructor
                ///
                stoch_sia_t(    std::size_t epochs,
                                std::size_t epoch_size,
                                tscalar alpha0,
                                tscalar decay,
                                const topulog& ulog = topulog())
                        :       m_param(epochs, epoch_size, alpha0, decay, ulog)
                {
                }

                ///
                /// \brief minimize starting from the initial guess x0
                ///
                tstate operator()(const tproblem& problem, const tvector& x0) const
                {
                        assert(problem.size() == x0.size());

                        // current state
                        tstate cstate(problem, x0);

                        // best state
                        best_state_t<tstate> bstate(cstate);

                        // running-weighted-averaged parameters
                        average_vector_t<tvector> xavg(x0.size());

                        for (std::size_t e = 0, k = 1; e < m_param.m_epochs; e ++)
                        {
                                for (std::size_t i = 0; i < m_param.m_epoch_size; i ++, k ++)
                                {
                                        // learning rate
                                        const tscalar alpha = m_param.alpha(k);

                                        // descent direction
                                        cstate.d = -cstate.g;

                                        // update solution
                                        cstate.update(problem, alpha);

                                        xavg.update(cstate.x);
                                }

                                const tvector cx = cstate.x;
                                cstate.update(problem, xavg.value());   // NB: to correctly log the current parameters!
                                m_param.ulog(cstate);
                                bstate.update(cstate);
                                cstate.update(problem, cx);             // revert it
                        }

                        // OK
                        return bstate.get();
                }

                // attributes
                param_t         m_param;
        };
}

