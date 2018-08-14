#pragma once

#include "tuner.h"
#include "core/json.h"
#include "core/factory.h"
#include "solver_state.h"

namespace nano
{
        class solver_t;
        using solver_factory_t = factory_t<solver_t>;
        using rsolver_t = solver_factory_t::trobject;

        NANO_PUBLIC solver_factory_t& get_solvers();

        ///
        /// \brief generic (batch) optimization algorithm typically using an adaptive line-search method.
        ///
        class NANO_PUBLIC solver_t : public json_configurable_t
        {
        public:

                ///
                /// logging operator: op(state), returns false if the optimization should stop
                ///
                using logger_t = std::function<bool(const solver_state_t&)>;

                ///
                /// \brief minimize the given function starting from the initial point x0
                ///
                virtual solver_state_t minimize(
                        const size_t max_iterations, const scalar_t epsilon,
                        const function_t&, const vector_t& x0,
                        const logger_t& logger = logger_t()) const = 0;

                ///
                /// \brief generate the hyper-parameters to tune
                ///
                virtual tuner_t tuner() const = 0;

        protected:

                static auto log(const logger_t& logger, const solver_state_t& state)
                {
                        return !logger ? true : logger(state);
                }

                ///
                /// \brief batch optimization loop running until:
                ///     - convergence is achieved (critical point, possiblly a local/global minima) or
                ///     - the maximum number of iterations is reached or
                ///     - the user canceled the optimization (using the logging function) or
                ///     - the solver failed (e.g. line-search failed)
                ///
                template <typename tsolver>
                static auto loop(
                        const function_t& function, const vector_t& x0,
                        const size_t max_iterations, const scalar_t epsilon, const logger_t& logger,
                        const tsolver& solver)
                {
                        assert(function.size() == x0.size());

                        auto state = solver_state_t{function, x0};

                        for (size_t i = 0; i < max_iterations; i ++)
                        {
                                const auto step_ok = solver(state, i) && state;
                                const auto converged = state.converged(epsilon);

                                if (converged || !step_ok)
                                {
                                        // either converged or failed
                                        state.m_status = converged ?
                                                solver_state_t::status::converged :
                                                solver_state_t::status::failed;
                                        log(logger, state);
                                        break;
                                }
                                else if (!log(logger, state))
                                {
                                        // stopping was requested
                                        state.m_status = solver_state_t::status::stopped;
                                        break;
                                }
                        }

                        return state;
                }
        };
}
