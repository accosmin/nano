#pragma once

#include "arch.h"
#include "optimizer.h"
#include "min/minimize.hpp"

namespace ncv
{
        ///
        /// \brief fixed list of initial learning rates to tune for the given stochastic method
        ///
        NANOCV_PUBLIC std::vector<opt_scalar_t> tunable_alphas(min::stoch_optimizer);

        ///
        /// \brief fixed list of decay rates to tune for the given stochastic method
        ///
        NANOCV_PUBLIC std::vector<opt_scalar_t> tunable_decays(min::stoch_optimizer);

        ///
        /// \brief tune the parameters for the given stochastic method
        ///
        NANOCV_PUBLIC void tune_stochastic(
                const opt_problem_t& problem, const opt_vector_t& x0,
                min::stoch_optimizer optimizer, opt_size_t epoch_size,
                opt_scalar_t& alpha0, opt_scalar_t& decay);
}
