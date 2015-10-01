#pragma once

#include "stoch_params.hpp"
#include "average_vector.hpp"

namespace min
{
        ///
        /// \brief stochastic gradient average gradient (descent)
        ///     see "Minimizing Finite Sums with the Stochastic Average Gradient",
        ///     by Mark Schmidth, Nicolas Le Roux, Francis Bach
        ///
        template
        <
                typename tproblem               ///< optimization problem
        >
        struct stoch_sga_t
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
                stoch_sga_t(    tsize epochs,
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

                        // running-weighted-averaged gradient
                        average_vector_t<tscalar, tvector> gavg(x0.size());

                        for (tsize e = 0, k = 1; e < m_param.m_epochs; e ++)
                        {
                                for (tsize i = 0; i < m_param.m_epoch_size; i ++, k ++)
                                {
                                        // learning rate
                                        const tscalar alpha = m_param.alpha(k);

                                        // descent direction
                                        gavg.update(cstate.g, m_param.weight(k));
                                        cstate.d = -gavg.value();

                                        // update solution
                                        cstate.update(problem, alpha);
                                }

                                m_param.ulog(cstate);
                        }

                        // OK
                        return cstate;
                }

                // attributes
                param_t         m_param;
        };
}

