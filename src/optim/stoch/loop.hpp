#pragma once

#include "params.h"
#include "math/tune.hpp"
#include "math/momentum.hpp"

namespace nano
{
        ///
        /// \brief hyper-parameter tuning for stochastic optimizers.
        ///
        inline auto make_alpha0s()
        {
                return make_finite_space(scalar_t(1e-3), scalar_t(1e-2), scalar_t(1e-1));
        }

        inline auto make_decays()
        {
                return make_log10_space(scalar_t(-3.0), scalar_t(-1.0), scalar_t(0.2));
        }

        inline auto make_momenta()
        {
                return make_log10_space(std::log10(scalar_t(0.1)), std::log10(scalar_t(0.99)), scalar_t(0.2));
        }

        inline auto make_epsilons()
        {
                return make_finite_space(scalar_t(1e-4), scalar_t(1e-6), scalar_t(1e-8));
        }

        ///
        /// \brief stochastic optimization loop until:
        ///     - the maximum number of iterations/epochs is reached or
        ///     - the user canceled the optimization (using the logging function)
        /// NB: convergence to a critical point is not guaranteed in general.
        ///
        template
        <
                typename toptimizer     ///< optimization method
        >
        auto stoch_loop(
                const problem_t& problem,
                const stoch_params_t& params,
                const state_t& istate,
                const toptimizer& optimizer,
                const stoch_params_t::config_t& config)
        {
                // current state
                auto cstate = istate;

                // average state
                // - similar to average stochastic gradient descent, but using an exponential moving average
                auto astate = istate;

                const scalar_t momentum = scalar_t(0.95);
                momentum_vector_t<vector_t> xavg(momentum, istate.x.size());

                // best state
                auto bstate = istate;
                bstate.f = std::numeric_limits<scalar_t>::max();

                // for each epoch ...
                for (std::size_t e = 0; e < params.m_epochs; ++ e)
                {
                        // for each iteration ...
                        for (std::size_t i = 0; i < params.m_epoch_size; ++ i)
                        {
                                optimizer(cstate);
                                xavg.update(cstate.x);
                        }

                        // log the current state & check the stopping criteria
                        astate.update(problem, xavg.value());
                        if (params.tuning())
                        {
                                astate.f = params.tlog(astate, config);
                        }
                        else if (!params.ulog(astate, config))
                        {
                                break;
                        }

                        // update the best state
                        bstate.update(astate);
                }

                // OK
                return bstate;
        }
}

