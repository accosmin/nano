#pragma once

#include "types.h"
#include "optim/state.h"
#include <functional>

namespace nano
{
        ///
        /// \brief common parameters for batch optimization
        ///
        struct batch_params_t
        {
                /// logging operator: op(state), returns false if the optimization should stop
                using opulog_t = std::function<bool(const state_t&)>;

                ///
                /// \brief constructor
                ///
                batch_params_t( const std::size_t max_iterations,
                                const scalar_t epsilon,
                                const ls_initializer lsinit,
                                const ls_strategy lsstrat,
                                const std::size_t hsize,
                                const opulog_t& ulog = opulog_t()) :
                        m_ulog(ulog),
                        m_max_iterations(max_iterations),
                        m_epsilon(epsilon),
                        m_ls_initializer(lsinit),
                        m_ls_strategy(lsstrat),
                        m_hsize(hsize)
                {
                }

                ///
                /// \brief log the current optimization state
                ///
                bool ulog(const state_t& state) const
                {
                        return m_ulog ? m_ulog(state) : true;
                }

                // attributes
                opulog_t        m_ulog;                 ///< update log: (the current_state_after_each_epoch)
                std::size_t     m_max_iterations;       ///< maximum number of iterations
                scalar_t        m_epsilon;              ///< convergence precision
                ls_initializer  m_ls_initializer;       ///< line-search step length initialization strategy
                ls_strategy     m_ls_strategy;          ///< line-search step length selection strategy
                std::size_t     m_hsize;                ///< number of previous iterations to use (if applicable)
        };
}

