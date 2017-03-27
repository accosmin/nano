#pragma once

#include "state.h"
#include <functional>

namespace nano
{
        ///
        /// \brief common parameters for stochastic optimization
        ///
        struct stoch_params_t
        {
                /// logging operator: op(state, configuration), returns true if the optimization should stop
                using opulog_t = std::function<bool(const state_t&, const string_t&)>;

                /// tunning operator: op(state, configuration)
                using optlog_t = std::function<void(const state_t&, const string_t&)>;

                ///
                /// \brief constructor
                ///
                stoch_params_t(
                        const size_t max_epochs,
                        const size_t epoch_size,
                        const scalar_t epsilon,
                        const opulog_t& ulog = opulog_t(),
                        const optlog_t& tlog = optlog_t()) :
                        m_max_epochs(max_epochs),
                        m_epoch_size(epoch_size),
                        m_epsilon(epsilon),
                        m_ulog(ulog),
                        m_tlog(tlog)
                {
                }

                ///
                /// \brief tuning parameters (use a single epoch, disable logging)
                ///
                auto tunable() const
                {
                        return stoch_params_t{1, m_epoch_size, m_epsilon, nullptr, m_tlog};
                }

                ///
                /// \brief optimization parameters (disable tuning)
                ///
                auto tuned() const
                {
                        return stoch_params_t{m_max_epochs, m_epoch_size, m_epsilon, m_ulog, nullptr};
                }

                ///
                /// \brief log the current optimization state
                ///
                bool ulog(const state_t& state, const string_t& config) const
                {
                        return m_ulog ? m_ulog(state, config) : true;
                }

                ///
                /// \brief evaluate the current optimization state for tuning
                ///
                void tlog(const state_t& state, const string_t& config) const
                {
                        if (m_tlog)
                        {
                                m_tlog(state, config);
                        }
                }

                // attributes
                size_t          m_max_epochs;           ///< number of epochs
                size_t          m_epoch_size;           ///< epoch size in number of iterations
                scalar_t        m_epsilon;              ///< convergence precision
                opulog_t        m_ulog;                 ///< update log: (the current_state_after_each_epoch)
                optlog_t        m_tlog;                 ///< tuning log: (the current_state_after_each_epoch)
        };
}
