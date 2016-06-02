#pragma once

#include <vector>
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

                /// tuning operator: op(state, configuration), returns the value associated with this configuration
                using optlog_t = std::function<scalar_t(const state_t&, const config_t&)>;

                enum class state
                {
                        tuning,
                        optimization
                };

                ///
                /// \brief constructor
                ///
                stoch_params_t( const std::size_t epochs,
                                const std::size_t epoch_size,
                                const opulog_t& ulog = opulog_t(),
                                const optlog_t& tlog = optlog_t(),
                                const state st = state::optimization) :
                        m_epochs(epochs),
                        m_epoch_size(epoch_size),
                        m_ulog(ulog),
                        m_tlog(tlog),
                        m_state(st)
                {
                }

                ///
                /// \brief construct the associated set of parameters suitable for tuning
                ///
                stoch_params_t tunable() const
                {
                        return { std::size_t(1), m_epoch_size, nullptr, m_tlog, state::tuning };
                }

                ///
                /// \brief check if in tuning state
                ///
                bool tuning() const
                {
                        return m_state == state::tuning;
                }

                ///
                /// \brief log the current optimization state
                ///
                bool ulog(const state_t& state, const config_t& config) const
                {
                        return m_ulog ? m_ulog(state, config) : true;
                }

                ///
                /// \brief log the current tuning state
                ///
                scalar_t tlog(const state_t& state, const config_t& config) const
                {
                        return m_tlog ? m_tlog(state, config) : state.f;
                }

                // attributes
                std::size_t     m_epochs;               ///< number of epochs
                std::size_t     m_epoch_size;           ///< epoch size in number of iterations
                opulog_t        m_ulog;                 ///< update log: (the current_state_after_each_epoch)
                optlog_t        m_tlog;                 ///< tuning log: (the current_state_after_first_epoch)
                state           m_state;                ///
        };
}
