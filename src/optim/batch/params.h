#pragma once

#include "types.h"
#include "optim/state.h"
#include <functional>

namespace nano
{
        ///
        /// \brief common parameters for batch optimization
        ///
        struct NANO_PUBLIC batch_params_t
        {
                /// logging operator: op(state), returns false if the optimization should stop
                using opulog_t = std::function<bool(const state_t&)>;

                ///
                /// \brief constructor (custom line-search)
                ///
                batch_params_t( const std::size_t max_iterations,
                                const scalar_t epsilon,
                                const batch_optimizer optimizer,
                                const ls_initializer lsinit,
                                const ls_strategy lsstrat,
                                const opulog_t& ulog = opulog_t(),
                                const std::size_t lbfgs_hsize = 6,
                                const scalar_t cgd_orthotest = scalar_t(0.1));

                ///
                /// \brief constructor (default line-search)
                ///
                batch_params_t( const std::size_t max_iterations,
                                const scalar_t epsilon,
                                const batch_optimizer optimizer,
                                const opulog_t& ulog = opulog_t(),
                                const std::size_t lbfgs_hsize = 6,
                                const scalar_t cgd_orthotest = scalar_t(0.1));

                ///
                /// \brief log the current optimization state
                ///
                bool ulog(const state_t& state) const
                {
                        return m_ulog ? m_ulog(state) : true;
                }

                // attributes
                opulog_t        m_ulog;                 ///< logging
                std::size_t     m_max_iterations;       ///< maximum number of iterations
                scalar_t        m_epsilon;              ///< convergence precision
                batch_optimizer m_optimizer;            ///< optimization algorithm
                ls_initializer  m_ls_initializer;       ///< line-search step length initialization strategy
                ls_strategy     m_ls_strategy;          ///< line-search step length selection strategy
                std::size_t     m_lbfgs_hsize;          ///< history size (for LBFGS)
                scalar_t        m_cgd_orthotest;        ///< orthogonality test (for CGD)
        };
}

