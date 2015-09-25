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
                const vector_t& x0,
                min::batch_optimizer, size_t iterations, scalar_t epsilon,
                size_t history_size = 6);

        ///
        /// \brief batch optimization (can detail the line-search parameters)
        ///
        NANOCV_PUBLIC opt_state_t minimize(
                const opt_problem_t& problem,
                const opt_opulog_t& fn_ulog,
                const vector_t& x0,
                min::batch_optimizer, size_t iterations, scalar_t epsilon,
                min::ls_initializer,
                min::ls_strategy,
                size_t history_size = 6);

        ///
        /// \brief stochastic optimization
        ///
        NANOCV_PUBLIC opt_state_t minimize(
                const opt_problem_t& problem,
                const opt_opulog_t& fn_ulog,
                const vector_t& x0,
                min::stoch_optimizer, size_t epochs, size_t epoch_size, scalar_t alpha0, scalar_t decay = 0.50);
}
