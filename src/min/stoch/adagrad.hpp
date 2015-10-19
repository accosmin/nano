#pragma once

#include "params.hpp"
#include "best_state.hpp"
#include "average_vector.hpp"

namespace min
{
        ///
        /// \brief stochastic AdaGrad
        ///     see "Adaptive subgradient methods for online learning and stochastic optimization"
        ///     by J. C. Duchi, E. Hazan, and Y. Singer
        ///
        ///     see http://xcorr.net/2014/01/23/adagrad-eliminating-learning-rates-in-stochastic-gradient-descent/
        ///
        template
        <
                typename tproblem               ///< optimization problem
        >
        struct stoch_adagrad_t
        {
                using param_t = stoch_params_t<tproblem>;
                using tstate = typename param_t::tstate;
                using tscalar = typename param_t::tscalar;
                using tvector = typename param_t::tvector;
                using topulog = typename param_t::topulog;

                ///
                /// \brief constructor
                ///
                stoch_adagrad_t(std::size_t epochs,
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
                        assert(problem.size() == static_cast<std::size_t>(x0.size()));

                        // current state
                        tstate cstate(problem, x0);

                        // best state
                        best_state_t<tstate> bstate(cstate);

                        // running-weighted-averaged-per-dimension-squared gradient
                        average_vector_t<tscalar, tvector> gavg(x0.size());

                        for (std::size_t e = 0, k = 1; e < m_param.m_epochs; e ++)
                        {
                                for (std::size_t i = 0; i < m_param.m_epoch_size; i ++, k ++)
                                {
                                        // learning rate
                                        const tscalar alpha = m_param.m_alpha0;

                                        // descent direction
                                        gavg.update(cstate.g.array().square(), m_param.weight(k));

                                        cstate.d = -cstate.g.array() /
                                                   (m_param.m_epsilon + gavg.value().array()).sqrt();

                                        // update solution
                                        cstate.update(problem, alpha);
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

