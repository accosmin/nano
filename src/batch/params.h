#pragma once

#include "types.h"
#include "solver_state.h"
#include <functional>

namespace nano
{
        ///
        /// \brief common parameters for batch optimization
        ///
        struct batch_params_t
        {
                /// logging operator: op(state), returns false if the optimization should stop
                using opulog_t = std::function<bool(const solver_state_t&)>;

                ///
                /// \brief constructor
                ///
                batch_params_t( const size_t max_iterations,
                                const scalar_t epsilon,
                                const opulog_t& ulog = opulog_t()) :
                        m_max_iterations(max_iterations),
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
                size_t          m_max_iterations;       ///< maximum number of iterations
                scalar_t        m_epsilon;              ///< convergence precision
                opulog_t        m_ulog;                 ///< logging
        };
}
