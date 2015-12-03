#pragma once

#include "params.hpp"
#include "best_state.hpp"
#include "math/momentum.hpp"

namespace math
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
                using param_t = stoch_params_t<tproblem>;
                using tstate = typename param_t::tstate;
                using tscalar = typename param_t::tscalar;
                using tvector = typename param_t::tvector;
                using topulog = typename param_t::topulog;

                ///
                /// \brief constructor
                ///
                explicit stoch_adadelta_t(const param_t& param) : m_param(param)
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

                        // running-averaged-per-dimension-squared gradient
                        momentum_vector_t<tvector> gavg(m_param.m_momentum, tvector::Zero(x0.size()));

                        // running-averaged-per-dimension-squared step updates
                        momentum_vector_t<tvector> davg(m_param.m_momentum, tvector::Zero(x0.size()));

                        for (std::size_t e = 0, k = 1; e < m_param.m_epochs; ++ e)
                        {
                                for (std::size_t i = 0; i < m_param.m_epoch_size; ++ i, ++ k) 
                                {
                                        // descent direction
                                        gavg.update(cstate.g.array().square());

                                        cstate.d = -cstate.g.array() *
                                                   (m_param.m_epsilon + davg.value().array()).sqrt() /
                                                   (m_param.m_epsilon + gavg.value().array()).sqrt();

                                        davg.update(cstate.d.array().square());

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

