#pragma once

#include "tuner.h"
#include "factory.h"
#include "function.h"
#include "stoch/params.h"
#include "configurable.h"

namespace nano
{
        class stoch_solver_t;
        using stoch_solver_factory_t = factory_t<stoch_solver_t>;
        using rstoch_solver_t = stoch_solver_factory_t::trobject;

        NANO_PUBLIC stoch_solver_factory_t& get_stoch_solvers();

        ///
        /// \brief generic stochastic solver that used an apriori learning-rate schedule.
        /// NB: all its hyper-parameters are tuned automatically.
        ///
        class NANO_PUBLIC stoch_solver_t : public configurable_t
        {
        public:

                ///
                /// \brief generate the hyper-parameters to tune.
                ///
                virtual tuner_t configs() const = 0;

                ///
                /// \brief tune its hyper-parameters to minimize the given function.
                ///
                solver_state_t tune(const stoch_params_t&, const function_t&, const vector_t& x0,
                        const size_t trials_per_parameter = 10);

                ///
                /// \brief minimize starting from the initial point x0.
                ///
                virtual solver_state_t minimize(const stoch_params_t&, const function_t&, const vector_t& x0) const = 0;

        protected:

                ///
                /// \brief stochastic optimization loop until:
                ///     - convergence is achieved (critical point, possiblly a local/global minima) or
                ///     - the maximum number of epochs is reached or
                ///     - the user canceled the optimization (using the logging function)
                /// NB: convergence to a critical point is not guaranteed in general.
                ///
                template
                <
                        typename tsolver,       ///< optimization algorithm: update the current state
                        typename tsnapshot      ///< snapshot at the end of an epoch: update the final state
                >
                static auto loop(
                        const stoch_params_t& param,
                        const function_t& function,
                        const vector_t& x0,
                        const tsolver& solver,
                        const tsnapshot& snapshot)
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
                                        solver(cstate, fstate);
                                }

                                // check divergence
                                if (!cstate)
                                {
                                        fstate.m_status = opt_status::failed;
                                        break;
                                }

                                // check convergence (using the full gradient)
                                const auto prevf = fstate.f;
                                snapshot(cstate, fstate);
                                if (fstate.converged(param.m_epsilon))
                                {
                                        fstate.m_status = opt_status::converged;
                                        param.ulog(fstate);
                                        break;
                                }

                                // check if the function value actually decreases (e.g. parameters need more tuning)
                                else if (prevf < fstate.f + param.m_epsilon)
                                {
                                        fstate.m_status = opt_status::failed;
                                        break;
                                }

                                // log the current state & check the stopping criteria
                                if (!param.ulog(fstate))
                                {
                                        fstate.m_status = opt_status::stopped;
                                        break;
                                }
                        }

                        // OK
                        return fstate;
                }
        };
}
