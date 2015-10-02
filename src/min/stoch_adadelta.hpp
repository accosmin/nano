#pragma once

#include "best_state.hpp"
#include "stoch_params.hpp"
#include "average_vector.hpp"

namespace min
{
        ///
        /// \brief stochastic AdaDelta,
        ///     see "ADADELTA: An Adaptive Learning Rate Method", by Matthew D. Zeiler
        ///
        template
        <
                typename tproblem               ///< optimization problem
        >
        struct stoch_adadelta_t
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
                stoch_adadelta_t(tsize epochs,
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

                        // running-weighted-averaged-per-dimension-squared gradient
                        average_vector_t<tscalar, tvector> gavg(x0.size());

                        // running-weighted-averaged-per-dimension-squared step updates
                        average_vector_t<tscalar, tvector> davg(x0.size());

                        for (tsize e = 0, k = 1; e < m_param.m_epochs; e ++)
                        {
                                for (tsize i = 0; i < m_param.m_epoch_size; i ++, k ++)
                                {
                                        // descent direction
                                        gavg.update(cstate.g.array().square(), m_param.weight(k));

                                        cstate.d = -cstate.g.array() *
                                                   (m_param.m_epsilon + davg.value().array()).sqrt() /
                                                   (m_param.m_epsilon + gavg.value().array()).sqrt();

                                        davg.update(cstate.d.array().square(), m_param.weight(k));

                                        // update solution
                                        cstate.update(problem, tscalar(1));
                                }

                                m_param.ulog(cstate);
                                bstate.update(cstate);
                        }

                        // OK
                        return bstate.get();
                }

                // attributes
                param_t         m_param;
        };
}

