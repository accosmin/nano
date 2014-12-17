#pragma once

#include "params.hpp"

namespace ncv
{
        namespace optimize
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
                        typedef typename tproblem::tscalar      tscalar;
                        typedef typename tproblem::tsize        tsize;
                        typedef typename tproblem::tvector      tvector;
                        typedef typename tproblem::tstate       tstate;         ///< optimization state
                        typedef typename tproblem::twlog        twlog;
                        typedef typename tproblem::telog        telog;
                        typedef typename tproblem::tulog        tulog;

                        ///
                        /// \brief constructor
                        ///
                        batch_params_t( tsize max_iterations,
                                        tscalar epsilon,
                                        const twlog& wlog = twlog(),
                                        const telog& elog = telog(),
                                        const tulog& ulog = tulog())
                                :       params_t<tproblem>(wlog, elog, ulog),
                                        m_max_iterations(max_iterations),
                                        m_epsilon(epsilon)
                        {
                        }

                        ///
                        /// \brief destructor
                        ///
                        virtual ~batch_params_t()
                        {
                        }

                        ///
                        /// \brief change parameters
                        ///
                        void set_max_iterations(tsize max_iterations) { m_max_iterations = max_iterations; }
                        void set_epsilon(tscalar epsilon) { m_epsilon = epsilon; }

                        tsize           m_max_iterations;       ///< maximum number of iterations
                        tscalar         m_epsilon;              ///< convergence precision
                };
        }
}

