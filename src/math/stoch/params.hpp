#pragma once

#include <cstddef>

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

                /// logging operator: op(state), returns false if the optimization should stop
                using topulog = std::function<bool(const tstate&)>;

                ///
                /// \brief constructor
                ///
                stoch_params_t( const std::size_t epochs,
                                const std::size_t epoch_size,
                                const topulog& ulog = topulog())
                        :       m_epochs(epochs),
                                m_epoch_size(epoch_size),
                                m_ulog(ulog)
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
                bool ulog(const tstate& state) const
                {
                        return m_ulog ? m_ulog(state) : true;
                }

                // attributes
                std::size_t             m_epochs;       ///< number of epochs
                std::size_t             m_epoch_size;   ///< epoch size in number of iterations
                topulog                 m_ulog;         ///< update log: (the current_state_after_each_epoch)
        };
}
