#pragma once

#include "best_state.hpp"
#include "stoch_params.hpp"
#include "average_vector.hpp"

namespace min
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
                typedef stoch_params_t<tproblem>        param_t;
                typedef typename param_t::tscalar       tscalar;
                typedef typename param_t::tsize         tsize;
                typedef typename param_t::tvector       tvector;
                typedef typename param_t::tstate        tstate;
                typedef typename param_t::top_ulog      top_ulog;

                ///
                /// \brief constructor
                ///
                stoch_sia_t(    tsize epochs,
                                tsize epoch_size,
                                tscalar alpha0,
                                tscalar decay,
                                const top_ulog& ulog = top_ulog())
                        :       m_param(epochs, epoch_size, alpha0, decay, ulog)
                {
                }

                ///
                /// \brief minimize starting from the initial guess x0
                ///
                tstate operator()(const tproblem& problem, const tvector& x0) const
                {
                        assert(problem.size() == static_cast<tsize>(x0.size()));

                        // current state
                        tstate cstate(problem, x0);

                        // best state
                        best_state_t<tstate> bstate(cstate);

                        // running-weighted-averaged parameters
                        average_vector_t<tscalar, tvector> xavg(x0.size());

                        for (tsize e = 0, k = 1; e < m_param.m_epochs; e ++)
                        {
                                for (tsize i = 0; i < m_param.m_epoch_size; i ++, k ++)
                                {
                                        // learning rate
                                        const tscalar alpha = m_param.alpha(k);

                                        // descent direction
                                        cstate.d = -cstate.g;

                                        // update solution
                                        cstate.update(problem, alpha);

                                        xavg.update(cstate.x, m_param.weight(k));
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

