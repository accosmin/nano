#pragma once

#include <vector>
#include <utility>
#include <functional>

namespace nano
{
        ///
        /// \brief common parameters for stochastic optimization
        ///
        template
        <
                typename tproblem                       ///< optimization problem
        >
        struct stoch_params_t
        {
                using tstate = typename tproblem::tstate;
                using tscalar = typename tproblem::tscalar;
                using tvector = typename tproblem::tvector;

                /// configuration: { hyper-parameter name, hyper-parameter value }+
                using tconfig = std::vector<std::pair<const char*, tscalar>>;

                /// logging operator: op(state, configuration), returns true if the optimization should stop
                using topulog = std::function<bool(const tstate&, const tconfig&)>;

                /// tuning operator: op(state, configuration), returns the value associated with this configuration
                using toptlog = std::function<tscalar(const tstate&, const tconfig&)>;

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
                                const topulog& ulog = topulog(),
                                const toptlog& tlog = toptlog(),
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
                bool ulog(const tstate& state, const tconfig& config) const
                {
                        return m_ulog ? m_ulog(state, config) : true;
                }

                ///
                /// \brief log the current tuning state
                ///
                tscalar tlog(const tstate& state, const tconfig& config) const
                {
                        return m_tlog ? m_tlog(state, config) : state.f;
                }

                // attributes
                std::size_t             m_epochs;       ///< number of epochs
                std::size_t             m_epoch_size;   ///< epoch size in number of iterations
                topulog                 m_ulog;         ///< update log: (the current_state_after_each_epoch)
                toptlog                 m_tlog;         ///< tuning log: (the current_state_after_first_epoch)
                state                   m_state;        ///
        };
}
