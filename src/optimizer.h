#pragma once

#include "factory.h"
#include "configurable.h"
#include "optimization_state.h"

namespace nano
{
        class optimizer_t;
        using optimizer_factory_t = factory_t<optimizer_t>;
        using roptimizer_t = optimizer_factory_t::trobject;

        NANO_PUBLIC optimizer_factory_t& get_optimizers();

        ///
        /// \brief generic (batch) optimization algorithm typically using an adaptive line-search method.
        ///
        class NANO_PUBLIC optimizer_t : public configurable_t
        {
        public:

                ///
                /// logging operator: op(state), returns false if the optimization should stop
                ///
                using logger_t = std::function<bool(const solver_state_t&)>;

                ///
                /// \brief minimize the given function starting from the initial point x0
                ///
                virtual optimization_state_t_state_t minimize(
                        const size_t max_iterations, const scalar_t epsilon,
                        const function_t&, const vector_t& x0,
                        const logger_t& logger = logger_t()) const = 0;

        protected:

                static auto log(const logger_t& logger, const solver_state_t& state)
                {
                        return logger ? true : logger(state);
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

                        auto cstate = make_state(function, x0);

                        for (size_t i = 0; i < max_iterations; i ++)
                        {
                                const auto step_ok = solver(cstate, i) && cstate;
                                const auto converged = cstate.converged(epsilon);

                                if (converged || !step_ok)
                                {
                                        // either converged or failed
                                        cstate.m_status = step_ok ?
                                                optimization_state_t::status::converged :
                                                optimization_state_t::status::failed;
                                        log(logger, cstate);
                                        break;
                                }
                                else if (!log(logger, cstate))
                                {
                                        // stopping was requested
                                        cstate.m_status = optimization_state_t::status::stopped;
                                        break;
                                }
                        }

                        return cstate;
                }
        };
}
