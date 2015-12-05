#pragma once

#include "stoch.hpp"
#include "tune_fixed.hpp"
#include <vector>
#include <limits>

namespace math
{
        ///
        /// \brief fixed list of initial learning rates to tune for the given stochastic method
        ///
        template
        <
                typename tscalar
        >
        std::vector<tscalar> tunable_alphas(const math::stoch_optimizer optimizer)
        {
                switch (optimizer)
                {
                case math::stoch_optimizer::ADADELTA:
                        return { tscalar(0.0) };

                default:
                        return { tscalar(1e-6), tscalar(1e-5), tscalar(1e-4),
                                 tscalar(1e-3), tscalar(1e-2), tscalar(1e-1),
                                 tscalar(1e+0) };
                }
        }

        ///
        /// \brief fixed list of decay rates to tune for the given stochastic method
        ///
        template
        <
                typename tscalar
        >
        std::vector<tscalar> tunable_decays(const math::stoch_optimizer optimizer)
        {
                switch (optimizer)
                {
                case math::stoch_optimizer::AG:
                case math::stoch_optimizer::AGFR:
                case math::stoch_optimizer::AGGR:
                case math::stoch_optimizer::ADAGRAD:
                case math::stoch_optimizer::ADADELTA:
                        return { tscalar(1.00) };

                default:
                        return { tscalar(0.50), tscalar(0.75) };
                }
        }

         ///
        /// \brief fixed list of momentum rates to tune for the given stochastic method
        ///
        template
        <
                typename tscalar
        >
        std::vector<tscalar> tunable_moments(const math::stoch_optimizer optimizer)
        {
                switch (optimizer)
                {
                case math::stoch_optimizer::SGM:
                case math::stoch_optimizer::ADADELTA:
                        return { tscalar(0.50), tscalar(0.80), tscalar(0.90), tscalar(0.95), tscalar(0.99) };

                default:
                        return { tscalar(0.0) };
                }
        }

        ///
        /// \brief tune the parameters for the given stochastic method
        ///
        template
        <
                typename tproblem,
                typename tsize = typename tproblem::tsize,
                typename tscalar = typename tproblem::tscalar,
                typename tvector = typename tproblem::tvector
        >
        void tune_stochastic(
                const tproblem& problem, const tvector& x0,
                math::stoch_optimizer optimizer, tsize epoch_size,
                tscalar& best_alpha, tscalar& best_decay, tscalar& best_momentum)
        {
                const auto alphas = tunable_alphas<tscalar>(optimizer);
                const auto decays = tunable_decays<tscalar>(optimizer);
                const auto moments = tunable_moments<tscalar>(optimizer);

                const auto op = [&] (const tscalar alpha, const tscalar decay, const tscalar momentum)
                {
                        const auto state = math::minimize(problem, nullptr, x0, optimizer, 1, epoch_size, alpha, decay, momentum);
                        const auto valid = std::isfinite(state.f);
                        return valid ? state.f : std::numeric_limits<tscalar>::max();
                };

                const auto config = math::tune_fixed(op, alphas, decays, moments);
                best_alpha = std::get<1>(config);
                best_decay = std::get<2>(config);
                best_momentum = std::get<3>(config);
        }
}
