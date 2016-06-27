#pragma once

#include <utility>
#include <functional>
#include "types.h"
#include "optim/state.h"

namespace nano
{
        ///
        /// \brief common parameters for stochastic optimization
        ///
        struct stoch_params_t
        {
                /// configuration: { hyper-parameter name, hyper-parameter value }+
                using config_t = std::vector<std::pair<const char*, scalar_t>>;

                /// logging operator: op(state, configuration), returns true if the optimization should stop
                using opulog_t = std::function<bool(const state_t&, const config_t&)>;

                ///
                /// \brief constructor
                ///
                stoch_params_t(
                        const std::size_t epochs,
                        const std::size_t epoch_size,
                        const stoch_optimizer optimizer,
                        const opulog_t& ulog = opulog_t()) :
                        m_epochs(epochs),
                        m_epoch_size(epoch_size),
                        m_optimizer(optimizer),
                        m_ulog(ulog)
                {
                }

                ///
                /// \brief log the current optimization state
                ///
                bool ulog(const state_t& state, const config_t& config) const
                {
                        return m_ulog ? m_ulog(state, config) : true;
                }

                // attributes
                std::size_t     m_epochs;               ///< number of epochs
                std::size_t     m_epoch_size;           ///< epoch size in number of iterations
                stoch_optimizer m_optimizer;            ///< optimization algorithm
                opulog_t        m_ulog;                 ///< update log: (the current_state_after_each_epoch)
        };
}
