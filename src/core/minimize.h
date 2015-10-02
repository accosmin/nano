#pragma once

#include "arch.h"
#include "optimizer.h"

namespace ncv
{
        ///
        /// \brief batch optimization
        ///
        NANOCV_PUBLIC opt_state_t minimize(
                const opt_problem_t& problem,
                const opt_opulog_t& fn_ulog,
                const opt_vector_t& x0,
                min::batch_optimizer, size_t iterations, opt_scalar_t epsilon,
                size_t history_size = 6);

        ///
        /// \brief batch optimization (can detail the line-search parameters)
        ///
        NANOCV_PUBLIC opt_state_t minimize(
                const opt_problem_t& problem,
                const opt_opulog_t& fn_ulog,
                const opt_vector_t& x0,
                min::batch_optimizer, size_t iterations, opt_scalar_t epsilon,
                min::ls_initializer,
                min::ls_strategy,
                size_t history_size = 6);

        ///
        /// \brief stochastic optimization
        ///
        NANOCV_PUBLIC opt_state_t minimize(
                const opt_problem_t& problem,
                const opt_opulog_t& fn_ulog,
                const opt_vector_t& x0,
                min::stoch_optimizer, size_t epochs, size_t epoch_size, opt_scalar_t alpha0, opt_scalar_t decay = 0.50);

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
