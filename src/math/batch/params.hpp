#pragma once

#include <cstddef>
#include "math/lsearch_types.h"

namespace zob
{
        ///
        /// \brief common parameters for batch optimization
        ///
        template
        <
                typename tproblem                       ///< optimization problem
        >
        struct batch_params_t
        {
                using tstate = typename tproblem::tstate;
                using tscalar = typename tproblem::tscalar;
                using tvector = typename tproblem::tvector;

                /// logging operator: op(state), returns false if the optimization should stop
                using topulog = std::function<bool(const tstate&)>;

                ///
                /// \brief constructor
                ///
                batch_params_t( std::size_t max_iterations,
                                tscalar epsilon,
                                ls_initializer lsinit,
                                ls_strategy lsstrat,
                                std::size_t hsize,
                                const topulog& ulog = topulog())
                        :       m_ulog(ulog),
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
                bool ulog(const tstate& state) const
                {
                        return m_ulog ? m_ulog(state) : true;
                }

                // attributes
                topulog         m_ulog;                 ///< update log: (the current_state_after_each_epoch)
                std::size_t     m_max_iterations;       ///< maximum number of iterations
                tscalar         m_epsilon;              ///< convergence precision
                ls_initializer  m_ls_initializer;       ///< line-search step length initialization strategy
                ls_strategy     m_ls_strategy;          ///< line-search step length selection strategy
                std::size_t     m_hsize;                ///< number of previous iterations to use (if applicable)
        };
}

