#pragma once

#include "stoch_loop.hpp"
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
                explicit stoch_sia_t(const param_t& param) : m_param(param)
                {
                }

                ///
                /// \brief minimize starting from the initial guess x0
                ///
                tstate operator()(const tproblem& problem, const tvector& x0) const
                {
                        assert(problem.size() == x0.size());

                        // current state
                        tvector cx = x0;

                        // running-averaged parameters
                        average_vector_t<tvector> xavg(x0.size());

                        const auto op_iter = [&] (tstate& cstate, const std::size_t k)
                        {
                                // learning rate
                                const tscalar alpha = m_param.alpha(k);

                                // descent direction
                                cx -= alpha * cstate.g;

                                // update solution
                                cstate.update(problem, cx);

                                xavg.update(cx);
                        };

                        const auto op_epoch = [&] (tstate& cstate)
                        {
                                cstate.update(problem, xavg.value());
                        };

                        // OK, assembly the optimizer
                        return stoch_loop(m_param, tstate(problem, x0), op_iter, op_epoch);
                }

                // attributes
                param_t         m_param;
        };
}

