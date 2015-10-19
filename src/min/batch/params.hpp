#pragma once

#include "min/params.hpp"
#include "min/lsearch_types.h"
#include <cstddef>

namespace min
{
        ///
        /// \brief common parameters for batch optimization
        ///
        template
        <
                typename tproblem                       ///< optimization problem
        >
        struct batch_params_t : public params_t<tproblem>
        {
                using tstate = typename tproblem::tstate;
                using tscalar = typename tproblem::tscalar;
                using tvector = typename tproblem::tvector;
                using topulog = typename tproblem::topulog;

                ///
                /// \brief constructor
                ///
                batch_params_t( std::size_t max_iterations,
                                tscalar epsilon,
                                ls_initializer lsinit,
                                ls_strategy lsstrat,
                                const topulog& u = topulog())
                        :       params_t<tproblem>(u),
                                m_max_iterations(max_iterations),
                                m_epsilon(epsilon),
                                m_ls_initializer(lsinit),
                                m_ls_strategy(lsstrat)
                {
                }

                ///
                /// \brief destructor
                ///
                virtual ~batch_params_t()
                {
                }

                // attributes
                std::size_t     m_max_iterations;       ///< maximum number of iterations
                tscalar         m_epsilon;              ///< convergence precision

                ls_initializer  m_ls_initializer;       ///< line-search step length initialization strategy
                ls_strategy     m_ls_strategy;          ///< line-search step length selection strategy
        };
}

