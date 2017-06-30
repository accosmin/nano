#pragma once

#include "params.h"
#include "function.h"
#include "math/tune.h"

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
                return make_finite_space(scalar_t(0.50), scalar_t(0.75), scalar_t(0.90), scalar_t(0.95));
        }

        inline auto make_momenta()
        {
                return make_finite_space(scalar_t(0.10), scalar_t(0.50), scalar_t(0.90));
        }

        inline auto make_epsilons()
        {
                return make_finite_space(scalar_t(1e-6), scalar_t(1e-5), scalar_t(1e-4));
        }

        ///
        /// \brief tune the given stochastic optimizer.
        ///
        template <typename toptimizer, typename... tspaces>
        auto stoch_tune(const toptimizer* optimizer,
                const stoch_params_t& param, const function_t& function, const vector_t& x0, tspaces... spaces)
        {
                const auto tune_op = [&] (const auto... hypers)
                {
                        return optimizer->minimize(param.tunable(), function, x0, hypers...);
                };
                return nano::tune(tune_op, spaces...);
        }

        ///
        /// \brief stochastic optimization loop until:
        ///     - convergence is achieved (critical point, possiblly a local/global minima) or
        ///     - the maximum number of epochs is reached or
        ///     - the user canceled the optimization (using the logging function)
        /// NB: convergence to a critical point is not guaranteed in general.
        ///
        template
        <
                typename toptimizer,            ///< optimization algorithm: update the current state
                typename tsnapshot              ///< snapshot at the end of an epoch: update the final state
        >
        auto stoch_loop(
                const stoch_params_t& param,
                const function_t& function,
                const vector_t& x0,
                const toptimizer& optimizer,
                const tsnapshot& snapshot,
                const string_t& config)
        {
                assert(function.size() == x0.size());

                // current state
                auto cstate = make_stoch_state(function, x0);

                // final state
                auto fstate = make_state(function, x0);

                // for each epoch ...
                for (size_t e = 0; e < param.m_max_epochs; ++ e)
                {
                        // for each iteration ...
                        for (size_t i = 0; i < param.m_epoch_size && cstate; ++ i)
                        {
                                optimizer(cstate, fstate);
                        }

                        // check divergence
                        if (!cstate)
                        {
                                fstate.m_status = opt_status::failed;
                                break;
                        }

                        // check convergence (using the full gradient)
                        snapshot(cstate, fstate);
                        if (fstate.converged(param.m_epsilon))
                        {
                                fstate.m_status = opt_status::converged;
                                param.tlog(fstate, config);
                                param.ulog(fstate, config);
                                break;
                        }

                        // log the current state & check the stopping criteria
                        param.tlog(fstate, config);
                        if (!param.ulog(fstate, config))
                        {
                                fstate.m_status = opt_status::stopped;
                                break;
                        }
                }

                // OK
                return fstate;
        }
}
