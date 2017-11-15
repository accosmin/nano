#pragma once

#include "factory.h"
#include "function.h"
#include "math/tune.h"
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
                /// \brief serialization to JSON not needed
                ///
                json_reader_t& config(json_reader_t& reader) final { return reader; }
                json_writer_t& config(json_writer_t& writer) const final { return writer; }

                ///
                /// \brief minimize starting from the initial point x0.
                ///
                virtual solver_state_t minimize(const stoch_params_t&, const function_t&, const vector_t& x0) const = 0;

        protected:

                ///
                /// \brief hyper-parameter tuning for stochastic solvers.
                ///
                static auto make_alpha0s()
                {
                        return make_finite_space(scalar_t(1e-3), scalar_t(1e-2), scalar_t(1e-1));
                }

                static auto make_decays()
                {
                        return make_finite_space(scalar_t(0.50), scalar_t(0.75), scalar_t(0.90));
                }

                static auto make_momenta()
                {
                        return make_finite_space(scalar_t(0.10), scalar_t(0.50), scalar_t(0.90));
                }

                static auto make_epsilons()
                {
                        return make_finite_space(scalar_t(1e-6), scalar_t(1e-5), scalar_t(1e-4));
                }

                /// reference: http://www.cppsamples.com/common-tasks/apply-tuple-to-function.html
                /// \todo replace this with std::apply when moving to C++17
                template<typename F, typename Tuple, size_t ...S >
                static decltype(auto) apply_tuple_impl(F&& fn, Tuple&& t, std::index_sequence<S...>)
                {
                        return std::forward<F>(fn)(std::get<S>(std::forward<Tuple>(t))...);
                }

                template<typename F, typename Tuple>
                static decltype(auto) apply_from_tuple(F&& fn, Tuple&& t)
                {
                        std::size_t constexpr tSize = std::tuple_size<typename std::remove_reference<Tuple>::type>::value;
                        return apply_tuple_impl(std::forward<F>(fn), std::forward<Tuple>(t), std::make_index_sequence<tSize>());
                }

                ///
                /// \brief tune the given stochastic solver.
                ///
                template <typename tsolver, typename... tspaces>
                static auto tune(const tsolver* solver,
                        const stoch_params_t& param, const function_t& function, const vector_t& x0, tspaces... spaces)
                {
                        const auto tune_op = [&] (const auto... hypers)
                        {
                                return solver->minimize(param.tunable(), function, x0, hypers...);
                        };
                        const auto config = nano::tune(tune_op, spaces...);

                        const auto done_op = [&] (const auto... hypers)
                        {
                                return solver->minimize(param.tuned(), function, config.optimum().x, hypers...);
                        };
                        return apply_from_tuple(done_op, config.params());
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
                        typename tsolver,       ///< optimization algorithm: update the current state
                        typename tsnapshot      ///< snapshot at the end of an epoch: update the final state
                >
                static auto loop(
                        const stoch_params_t& param,
                        const function_t& function,
                        const vector_t& x0,
                        const tsolver& solver,
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
                                        solver(cstate, fstate);
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
        };
}
