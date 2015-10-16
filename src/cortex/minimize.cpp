#include "minimize.h"
#include "math/tune_fixed.hpp"

namespace ncv
{
        std::vector<opt_scalar_t> tunable_alphas(min::stoch_optimizer optimizer)
        {
                switch (optimizer)
                {
                case min::stoch_optimizer::ADADELTA:
                        return { 0.0 };

                default:
                        return { 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0 };
                }
        }

        std::vector<opt_scalar_t> tunable_decays(min::stoch_optimizer optimizer)
        {
                switch (optimizer)
                {
                case min::stoch_optimizer::AG:
                case min::stoch_optimizer::AGGR:
                case min::stoch_optimizer::ADAGRAD:
                case min::stoch_optimizer::ADADELTA:
                        return { 1.00 };

                default:
                        return { 0.50, 0.75 };
                }
        }

        void tune_stochastic(
                const opt_problem_t& problem, const opt_vector_t& x0,
                min::stoch_optimizer optimizer, opt_size_t epoch_size,
                opt_scalar_t& best_alpha0, opt_scalar_t& best_decay)
        {
                const auto alphas = tunable_alphas(optimizer);
                const auto decays = tunable_decays(optimizer);

                const auto op = [&] (const opt_scalar_t alpha, const opt_scalar_t decay)
                {
                        const auto state = min::minimize(problem, nullptr, x0, optimizer, 1, epoch_size, alpha, decay);
                        const auto valid = std::isfinite(state.f);
                        return valid ? state.f : std::numeric_limits<opt_scalar_t>::max();
                };

                const auto config = math::tune_fixed(op, alphas, decays);
                best_alpha0 = std::get<1>(config);
                best_decay = std::get<2>(config);
        }
}
	
