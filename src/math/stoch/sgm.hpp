#pragma once

#include "stoch_loop.hpp"
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

                        // running-averaged-per-dimension step updates
                        momentum_vector_t<tvector> davg(m_param.m_momentum, tvector::Zero(x0.size()));

                        const auto op_iter = [&] (tstate& cstate, const std::size_t k)
                        {
                                // learning rate
                                const tscalar alpha = m_param.alpha(k);

                                // descent direction
                                davg.update(cstate.g.array());
                                cstate.d = -davg.value();

                                // update solution
                                cstate.update(problem, alpha);
                        };

                        const auto op_epoch = [&] (tstate&)
                        {
                        };

                        // OK, assembly the optimizer
                        return stoch_loop(m_param, tstate(problem, x0), op_iter, op_epoch); 
                }

                // attributes
                param_t         m_param;
        };
}

