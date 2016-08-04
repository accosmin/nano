#pragma once

#include "state.h"
#include <utility>
#include <functional>

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

                /// tunning operator: op(state, configuration), returns the evaluation for a configuration
                using optlog_t = std::function<scalar_t(const state_t&, const config_t&)>;

                ///
                /// \brief constructor
                ///
                stoch_params_t(
                        const std::size_t epochs,
                        const std::size_t epoch_size,
                        const opulog_t& ulog = opulog_t(),
                        const optlog_t& tlog = optlog_t()) :
                        m_epochs(epochs),
                        m_epoch_size(epoch_size),
                        m_ulog(ulog),
                        m_tlog(tlog)
                {
                }

                ///
                /// \brief tuning parameters (use a single epoch, disable logging)
                ///
                auto tunable() const
                {
                        return stoch_params_t{1, m_epoch_size, nullptr, m_tlog};
                }

                ///
                /// \brief optimization parameters (disable tuning)
                ///
                auto tuned() const
                {
                        return stoch_params_t{m_epochs, m_epoch_size, m_ulog, nullptr};
                }

                ///
                /// \brief log the current optimization state
                ///
                bool ulog(const state_t& state, const config_t& config) const
                {
                        return m_ulog ? m_ulog(state, config) : true;
                }

                ///
                /// \brief evaluate the current optimization state for tuning
                ///
                scalar_t tlog(const state_t& state, const config_t& config) const
                {
                        return m_tlog ? m_tlog(state, config) : state.f;
                }

                // attributes
                std::size_t     m_epochs;               ///< number of epochs
                std::size_t     m_epoch_size;           ///< epoch size in number of iterations
                opulog_t        m_ulog;                 ///< update log: (the current_state_after_each_epoch)
                optlog_t        m_tlog;                 ///< tuning log: (the current_state_after_each_epoch)
        };
}
