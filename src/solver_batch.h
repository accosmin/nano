#pragma once

#include "factory.h"
#include "function.h"
#include "batch/params.h"
#include "configurable.h"

namespace nano
{
        class batch_solver_t;
        using batch_solver_factory_t = factory_t<batch_solver_t>;
        using rbatch_solver_t = batch_solver_factory_t::trobject;

        NANO_PUBLIC batch_solver_factory_t& get_batch_solvers();

        ///
        /// \brief generic batch solver that uses an adaptive line-search method.
        ///
        class NANO_PUBLIC batch_solver_t : public configurable_t
        {
        public:

                ///
                /// \brief minimize starting from the initial point x0
                ///
                virtual solver_state_t minimize(const batch_params_t&, const function_t&, const vector_t& x0) const = 0;

        protected:

                ///
                /// \brief batch optimization loop running until:
                ///     - convergence is achieved (critical point, possiblly a local/global minima) or
                ///     - the maximum number of iterations is reached or
                ///     - the user canceled the optimization (using the logging function) or
                ///     - the solver failed (e.g. line-search failed)
                ///
                template <typename tsolver>
                static auto loop(
                        const batch_params_t& params,
                        const function_t& function,
                        const vector_t& x0,
                        const tsolver& solver)
                {
                        assert(function.size() == x0.size());

                        // current state
                        auto cstate = make_state(function, x0);

                        // for each iteration ...
                        for (size_t i = 0; i < params.m_max_iterations; i ++)
                        {
                                if (!solver(cstate, i) || !cstate)
                                {
                                        cstate.m_status = cstate.converged(params.m_epsilon) ?
                                                opt_status::converged : opt_status::failed ;
                                        params.ulog(cstate);
                                        break;
                                }

                                // check convergence
                                if (cstate.converged(params.m_epsilon))
                                {
                                        cstate.m_status = opt_status::converged;
                                        params.ulog(cstate);
                                        break;
                                }

                                // log the current state & check the stopping criteria
                                if (!params.ulog(cstate))
                                {
                                        cstate.m_status = opt_status::stopped;
                                        break;
                                }
                        }

                        // OK
                        return cstate;
                }
        };
}
