#pragma once

#include "params.h"
#include "math/tune.h"
#include "math/momentum.h"

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
                return make_finite_space(scalar_t(0.01), scalar_t(0.1), scalar_t(0.2), scalar_t(0.5), scalar_t(0.75), scalar_t(1.0));
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
                // tuple unpacking for calling void function
                // http://stackoverflow.com/questions/10766112/c11-i-can-go-from-multiple-args-to-tuple-but-can-i-go-from-tuple-to-multiple

                template <typename F, typename Tuple, bool Done, int Total, int... N>
                struct call_impl
                {
                        static void call(F f, Tuple && t)
                        {
                                call_impl<F, Tuple, Total == 1 + sizeof...(N), Total, N..., sizeof...(N)
                                >::call(f, std::forward<Tuple>(t));
                        }
                };

                template <typename F, typename Tuple, int Total, int... N>
                struct call_impl<F, Tuple, true, Total, N...>
                {
                        static void call(F f, Tuple && t)
                        {
                                f(std::get<N>(std::forward<Tuple>(t))...);
                        }
                };

                template <typename F, typename Tuple>
                void call(F f, Tuple && t)
                {
                        typedef typename std::decay<Tuple>::type ttype;
                        detail::call_impl<F, Tuple, 0 == std::tuple_size<ttype>::value, std::tuple_size<ttype>::value
                        >::call(f, std::forward<Tuple>(t));
                }
        }

        ///
        /// \brief tune the given stochastic optimizer.
        ///
        template <typename toptimizer, typename... tspaces>
        auto stoch_tune(const toptimizer* optimizer,
                const stoch_params_t& param, const problem_t& problem, vector_t x0,
                tspaces... spaces)
        {
                const auto tune_op = [&] (const auto... hypers)
                {
                        return optimizer->minimize(param.tunable(), problem, x0, hypers...);
                };

                state_t state;
                const auto done_op = [&] (const auto... hypers)
                {
                        state = optimizer->minimize(param.tuned(), problem, x0, hypers...);
                };

                const auto config = nano::tune(tune_op, spaces...);
                x0 = config.optimum().x;
                detail::call(done_op, config.params());
                return state;
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

                // for each epoch ...
                for (size_t e = 0; e < params.m_epochs; ++ e)
                {
                        // for each iteration ...
                        for (size_t i = 0; i < params.m_epoch_size; ++ i)
                        {
                                optimizer(cstate);
                                xavg.update(cstate.x);
                        }

                        // check divergence
                        if (!astate || !cstate)
                        {
                                break;
                        }

                        // log the current state & check the stopping criteria
                        astate.update(problem, xavg.value());
                        astate.f = params.tlog(astate, config);
                        if (!params.ulog(astate, config))
                        {
                                astate.m_status = opt_status::stopped;
                                bstate.update(astate);
                                break;
                        }

                        // update the best state
                        bstate.update(astate);
                }

                // OK
                return bstate;
        }
}

