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
                struct params_t
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
                        params_t(       const twlog& wlog = twlog(),
                                        const telog& elog = telog(),
                                        const tulog& ulog = tulog())
                                :       m_wlog(wlog),
                                        m_elog(elog),
                                        m_ulog(ulog)
                        {
                        }

                        ///
                        /// \brief destructor
                        ///
                        virtual ~params_t()
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

                        ///
                        /// \brief change loggers
                        ///
                        void set_wlog(const twlog& wlog) { m_wlog = wlog; }
                        void set_elog(const telog& elog) { m_elog = elog; }
                        void set_ulog(const tulog& ulog) { m_ulog = ulog; }

                        twlog           m_wlog;                 ///< warning log: (string message)
                        telog           m_elog;                 ///< error log: (string message)
                        tulog           m_ulog;                 ///< update log: (tstate current_state_after_each_epoch)
                };
        }
}

