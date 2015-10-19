#pragma once

#include "stoch.hpp"
#include "tune_fixed.hpp"
#include <vector>
#include <limits>

namespace min
{
        ///
        /// \brief fixed list of initial learning rates to tune for the given stochastic method
        ///
        template
        <
                typename tscalar
        >
        std::vector<tscalar> tunable_alphas(const min::stoch_optimizer optimizer)
        {
                switch (optimizer)
                {
                case min::stoch_optimizer::ADADELTA:
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
        std::vector<tscalar> tunable_decays(const min::stoch_optimizer optimizer)
        {
                switch (optimizer)
                {
                case min::stoch_optimizer::AG:
                case min::stoch_optimizer::AGGR:
                case min::stoch_optimizer::ADAGRAD:
                case min::stoch_optimizer::ADADELTA:
                        return { tscalar(1.00) };

                default:
                        return { tscalar(0.50), tscalar(0.75) };
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
                min::stoch_optimizer optimizer, tsize epoch_size,
                tscalar& best_alpha, tscalar& best_decay)
        {
                const auto alphas = tunable_alphas<tscalar>(optimizer);
                const auto decays = tunable_decays<tscalar>(optimizer);

                const auto op = [&] (const tscalar alpha, const tscalar decay)
                {
                        const auto state = min::minimize(problem, nullptr, x0, optimizer, 1, epoch_size, alpha, decay);
                        const auto valid = std::isfinite(state.f);
                        return valid ? state.f : std::numeric_limits<tscalar>::max();
                };

                const auto config = min::tune_fixed(op, alphas, decays);
                best_alpha = std::get<1>(config);
                best_decay = std::get<2>(config);
        }
}
