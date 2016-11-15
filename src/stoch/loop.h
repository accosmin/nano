#pragma once

#include "params.h"
#include "math/tune.h"
#include <experimental/tuple>

namespace nano
{
        ///
        /// \brief hyper-parameter tuning for stochastic optimizers.
        ///
        inline auto make_alpha0s()
        {
                return make_finite_space(scalar_t(1e-3), scalar_t(1e-2), scalar_t(1e-1), scalar_t(1e+0));
        }

        inline auto make_decays()
        {
                return make_finite_space(scalar_t(0.1), scalar_t(0.2), scalar_t(0.5), scalar_t(0.75), scalar_t(1.0));
        }

        inline auto make_momenta()
        {
                return make_finite_space(scalar_t(0.5), scalar_t(0.9), scalar_t(0.95));
        }

        inline auto make_epsilons()
        {
                return make_finite_space(scalar_t(1e-4), scalar_t(1e-6));
        }

        ///
        /// \brief tune the given stochastic optimizer.
        ///
        template <typename toptimizer, typename... tspaces>
        auto stoch_tune(const toptimizer* optimizer,
                const stoch_params_t& param, const problem_t& problem, const vector_t& x0,
                tspaces... spaces)
        {
                const auto tune_op = [&] (const auto... hypers)
                {
                        return optimizer->minimize(param.tunable(), problem, x0, hypers...);
                };
                const auto config = nano::tune(tune_op, spaces...);

                const auto done_op = [&] (const auto... hypers)
                {
                        return optimizer->minimize(param.tuned(), problem, config.optimum().x, hypers...);
                };
                return std::experimental::apply(done_op, config.params());
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
                typename toptimizer     ///< optimization method
        >
        auto stoch_loop(
                const stoch_params_t& params,
                const state_t& istate,
                const toptimizer& optimizer,
                const stoch_params_t::config_t& config)
        {
                // current state
                auto cstate = istate;

                // for each epoch ...
                for (size_t e = 0; e < params.m_max_epochs; ++ e)
                {
                        // for each iteration ...
                        for (size_t i = 0; i < params.m_epoch_size; ++ i)
                        {
                                optimizer(cstate);
                        }

                        // check divergence
                        if (!cstate)
                        {
                                break;
                        }

                        /*// check convergence (using the full gradient)
                        cstate.update(problem, cstate.x);
                        if (cstate.converged(params.m_epsilon))
                        {
                                cstate.m_status = opt_status::converged;
                                break;
                        }*/

                        // log the current state & check the stopping criteria
                        params.tlog(cstate, config);
                        if (!params.ulog(cstate, config))
                        {
                                cstate.m_status = opt_status::stopped;
                                break;
                        }
                }

                // OK
                return cstate;
        }
}

