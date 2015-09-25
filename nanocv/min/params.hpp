#pragma once

namespace ncv
{
        namespace min
        {
                ///
                /// \brief common parameters for batch optimization
                ///
                template
                <
                        typename tproblem                       ///< optimization problem
                >
                class params_t
                {
                public:

                        typedef typename tproblem::tscalar      tscalar;
                        typedef typename tproblem::tsize        tsize;
                        typedef typename tproblem::tvector      tvector;
                        typedef typename tproblem::tstate       tstate;
                        typedef typename tproblem::tulog        tulog;

                        ///
                        /// \brief constructor
                        ///
                        params_t(const tulog& u = tulog())
                                :       m_ulog(u)
                        {
                        }

                        ///
                        /// \brief destructor
                        ///
                        virtual ~params_t()
                        {
                        }

                        ///
                        /// \brief log current optimization state
                        ///
                        bool ulog(const tstate& state) const
                        {
                                return m_ulog ? m_ulog(state) : true;
                        }

                        tulog           m_ulog;                 ///< update log: (tstate current_state_after_each_epoch)
                };
        }
}

