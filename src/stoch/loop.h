#pragma once

#include "params.h"
#include "problem.h"
#include "math/tune.h"

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
                return make_finite_space(scalar_t(0.75), scalar_t(0.90), scalar_t(1.0));
        }

        inline auto make_momenta()
        {
                return make_finite_space(scalar_t(0.5), scalar_t(0.9), scalar_t(0.95));
        }

        inline auto make_epsilons()
        {
                return make_finite_space(scalar_t(1e-4), scalar_t(1e-6));
        }

        namespace detail
        {
                /// reference: http://www.cppsamples.com/common-tasks/apply-tuple-to-function.html
                /// \todo replace this with std::apply when moving to C++17
                template<typename F, typename Tuple, size_t ...S >
                decltype(auto) apply_tuple_impl(F&& fn, Tuple&& t, std::index_sequence<S...>)
                {
                        return std::forward<F>(fn)(std::get<S>(std::forward<Tuple>(t))...);
                }

                template<typename F, typename Tuple>
                decltype(auto) apply_from_tuple(F&& fn, Tuple&& t)
                {
                        std::size_t constexpr tSize = std::tuple_size<typename std::remove_reference<Tuple>::type>::value;
                        return apply_tuple_impl(std::forward<F>(fn), std::forward<Tuple>(t), std::make_index_sequence<tSize>());
                }
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
                return detail::apply_from_tuple(done_op, config.params());
        }

        ///
        /// \brief stochastic optimization loop until:
        ///     - convergence is achieved (critical point, possiblly a local/global minima) or
        ///     - the maximum number of epochs is reached or
        ///     - the user canceled the optimization (using the logging function)
        /// NB: convergence to a critical point is not guaranteed in general.
        ///
        template <typename toptimizer>
        auto stoch_loop(
                const stoch_params_t& params,
                const problem_t& problem,
                const vector_t& x0,
                const toptimizer& optimizer,
                const stoch_params_t::config_t& config)
        {
                // current state
                state_t cstate(problem.size());
                cstate.stoch_update(problem, x0);

                // final state
                state_t fstate = cstate;

                // for each epoch ...
                for (size_t e = 0; e < params.m_max_epochs; ++ e)
                {
                        // for each iteration ...
                        for (size_t i = 0; i < params.m_epoch_size && cstate; ++ i)
                        {
                                optimizer(cstate);
                        }

                        // check divergence
                        if (!cstate)
                        {
                                fstate.m_status = opt_status::failed;
                                break;
                        }

                        // check convergence (using the full gradient)
                        fstate.update(problem, cstate.x);
                        if (fstate.converged(params.m_epsilon))
                        {
                                fstate.m_status = opt_status::converged;
                                params.tlog(fstate, config);
                                params.ulog(fstate, config);
                                break;
                        }

                        // log the current state & check the stopping criteria
                        params.tlog(fstate, config);
                        if (!params.ulog(fstate, config))
                        {
                                fstate.m_status = opt_status::stopped;
                                break;
                        }
                }

                // OK
                return fstate;
        }
}

