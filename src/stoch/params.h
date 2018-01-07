#pragma once

#include "solver_state.h"
#include <functional>

namespace nano
{
        ///
        /// \brief common parameters for stochastic optimization
        ///
        struct stoch_params_t
        {
                /// logging operator: op(state), returns true if the optimization should stop
                using opulog_t = std::function<bool(const solver_state_t&)>;

                ///
                /// \brief constructor
                ///
                stoch_params_t(
                        const size_t max_epochs,
                        const size_t epoch_size,
                        const scalar_t epsilon,
                        const opulog_t& ulog = opulog_t()) :
                        m_max_epochs(max_epochs),
                        m_epoch_size(epoch_size),
                        m_epsilon(epsilon),
                        m_ulog(ulog)
                {
                }

                ///
                /// \brief log the current optimization state
                ///
                bool ulog(const solver_state_t& state) const
                {
                        return m_ulog ? m_ulog(state) : true;
                }

                // attributes
                size_t          m_max_epochs;           ///< maximum number of epochs
                size_t          m_epoch_size;           ///< number of iterations per epoch
                scalar_t        m_epsilon;              ///< convergence precision
                opulog_t        m_ulog;                 ///< update log: (the current_state_after_each_epoch)
        };
}
