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
                /// logging operator: op(state, params), returns true if the optimization should stop
                using opulog_t = std::function<bool(const solver_state_t&, const string_t&)>;

                /// tunning operator: op(state, params)
                using optlog_t = std::function<void(const solver_state_t&, const string_t&)>;

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
                        m_tlog(tlog),
                        m_tune_max_epochs(1),
                        m_tune_epoch_size(epoch_size)
                {
                }

                ///
                /// \brief tuning parameters (use a single epoch, disable logging)
                ///
                auto tunable() const
                {
                        return stoch_params_t{m_tune_max_epochs, m_tune_epoch_size, m_epsilon, nullptr, m_tlog};
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
                bool ulog(const solver_state_t& state, const string_t& config) const
                {
                        return m_ulog ? m_ulog(state, config) : true;
                }

                ///
                /// \brief evaluate the current optimization state for tuning
                ///
                void tlog(const solver_state_t& state, const string_t& config) const
                {
                        if (m_tlog)
                        {
                                m_tlog(state, config);
                        }
                }

                // attributes
                size_t          m_max_epochs;           ///< maximum number of epochs
                size_t          m_epoch_size;           ///< number of iterations per epoch
                scalar_t        m_epsilon;              ///< convergence precision
                opulog_t        m_ulog;                 ///< update log: (the current_state_after_each_epoch)
                optlog_t        m_tlog;                 ///< tuning log: (the current_state_after_each_epoch)
                size_t          m_tune_max_epochs;      ///< maximum number of epochs (when tuning)
                size_t          m_tune_epoch_size;      ///< number of iterations per epoch (when tuning)
        };
}
