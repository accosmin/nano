#pragma once

#include "nanocv/optimizer.h"

namespace ncv
{
        ///
        /// \brief batch optimization
        ///
        NANOCV_PUBLIC opt_state_t minimize(
                const opt_opsize_t& fn_size,
                const opt_opfval_t& fn_fval,
                const opt_opgrad_t& fn_grad,
                const opt_opwlog_t& fn_wlog,
                const opt_opelog_t& fn_elog,
                const opt_opulog_t& fn_ulog,
                const vector_t& x0,
                optim::batch_optimizer, size_t iterations, scalar_t epsilon,
                size_t history_size = 6);

        ///
        /// \brief batch optimization (can detail the line-search parameters)
        ///
        NANOCV_PUBLIC opt_state_t minimize(
                const opt_opsize_t& fn_size,
                const opt_opfval_t& fn_fval,
                const opt_opgrad_t& fn_grad,
                const opt_opwlog_t& fn_wlog,
                const opt_opelog_t& fn_elog,
                const opt_opulog_t& fn_ulog,
                const vector_t& x0,
                optim::batch_optimizer, size_t iterations, scalar_t epsilon,
                optim::ls_initializer,
                optim::ls_strategy,
                size_t history_size = 6);

        ///
        /// \brief stochastic optimization
        ///
        NANOCV_PUBLIC opt_state_t minimize(
                const opt_opsize_t& fn_size,
                const opt_opfval_t& fn_fval,
                const opt_opgrad_t& fn_grad,
                const opt_opwlog_t& fn_wlog,
                const opt_opelog_t& fn_elog,
                const opt_opulog_t& fn_ulog,
                const vector_t& x0,
                optim::stoch_optimizer, size_t epochs, size_t epoch_size, scalar_t alpha0, scalar_t decay = 0.50);

        ///
        /// \brief warning logging operator
        ///
        opt_opwlog_t make_opwlog();

        ///
        /// \brief error logging operator
        ///
        opt_opelog_t make_opelog();
}
