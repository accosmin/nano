#pragma once

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
                struct batch_params
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
                        batch_params(
                                tsize max_iterations,
                                tscalar epsilon,
                                const twlog& wlog = twlog(),
                                const telog& elog = telog(),
                                const tulog& ulog = tulog())
                                :       m_max_iterations(max_iterations),
                                        m_epsilon(epsilon),
                                        m_wlog(wlog),
                                        m_elog(elog),
                                        m_ulog(ulog)
                        {
                        }

                        ///
                        /// \brief destructor
                        ///
                        virtual ~batch_params()
                        {
                        }

                        ///
                        /// \brief log warning message
                        ///
                        template
                        <
                                typename tstring
                        >
                        void wlog(const tstring& message) const
                        {
                                if (m_wlog) m_wlog(message);
                        }

                        ///
                        /// \brief log error message
                        ///
                        template
                        <
                                typename tstring
                        >
                        void elog(const tstring& message) const
                        {
                                if (m_elog) m_elog(message);
                        }

                        ///
                        /// \brief log current optimization state
                        ///
                        void ulog(const tstate& state) const
                        {
                                if (m_ulog) m_ulog(state);
                        }

                        tsize           m_max_iterations;       ///< maximum number of iterations
                        tscalar         m_epsilon;              ///< convergence precision
                        twlog           m_wlog;                 ///< warning log: (string message)
                        telog           m_elog;                 ///< error log: (string message)
                        tulog           m_ulog;                 ///< update log: (tstate)
                };
        }
}

