#pragma once

#include "params.hpp"
#include "best_state.hpp"
#include "math/momentum.hpp"

namespace math
{
        ///
        /// \brief stochastic gradient (descent) with momentum
        ///
        template
        <
                typename tproblem               ///< optimization problem
        >
        struct stoch_sgm_t
        {
                using param_t = stoch_params_t<tproblem>;
                using tstate = typename param_t::tstate;
                using tscalar = typename param_t::tscalar;
                using tvector = typename param_t::tvector;
                using topulog = typename param_t::topulog;

                ///
                /// \brief constructor
                ///
                explicit stoch_sgm_t(const param_t& param) : m_param(param)
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

                        // running-weighted-averaged-per-dimension step updates
                        momentum_vector_t<tvector> davg(m_param.m_momentum, tvector::Zero(x0.size()));

                        for (std::size_t e = 0, k = 1; e < m_param.m_epochs; ++ e)
                        {
                                for (std::size_t i = 0; i < m_param.m_epoch_size; ++ i, ++ k) 
                                {
                                        // learning rate
                                        const tscalar alpha = m_param.alpha(k);

                                        // descent direction
                                        davg.update(cstate.g.array());
                                        cstate.d = -davg.value();

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

