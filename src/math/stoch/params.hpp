#pragma once

#include <vector>
#include <cstddef>
#include <utility>

namespace math
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
                using tconfig = std::vector<std::pair<const char* const, tscalar>>;

                /// logging operator: op(state), returns false if the optimization should stop
                using topulog = std::function<bool(const tstate&, const tconfig&)>;

                /// tunning operator: op(state, configuration)
                using toptlog = std::function<void(const tstate&, const tconfig&)>;

                ///
                /// \brief constructor
                ///
                stoch_params_t( const std::size_t epochs,
                                const std::size_t epoch_size,
                                const topulog& ulog = topulog(),
                                const toptlog& tlog = toptlog())
                        :       m_epochs(epochs),
                                m_epoch_size(epoch_size),
                                m_ulog(ulog),
                                m_tlog(tlog)
                {
                }

                ///
                /// \brief construct the associated set of parameters suitable for tuning
                ///
                stoch_params_t tunable() const
                {
                        return { std::size_t(1), m_epoch_size, nullptr };
                }

                ///
                /// \brief log the current optimization state
                ///
                bool ulog(const tstate& state, const tconfig& config) const
                {
                        return m_ulog ? m_ulog(state, config) : true;
                }

                ///
                /// \brief log the optimization state associated with a hyper-parameters configuration
                ///
                void tlog(const tstate& state, const tconfig& config) const
                {
                        if (m_tlog)
                        {
                                m_tlog(state, config);
                        }
                }

                // attributes
                std::size_t             m_epochs;       ///< number of epochs
                std::size_t             m_epoch_size;   ///< epoch size in number of iterations
                topulog                 m_ulog;         ///< update log: (the current_state_after_each_epoch)
                toptlog                 m_tlog;         ///< tunning log: (the state, configuration)
        };
}
