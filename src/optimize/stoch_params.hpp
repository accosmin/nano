#pragma once

namespace ncv
{
        namespace optimize
        {
                ///
                /// \brief common parameters for stochastic optimization
                ///
                template
                <
                        typename tproblem                       ///< optimization problem
                >
                struct stoch_params
                {
                        typedef typename tproblem::tscalar      tscalar;
                        typedef typename tproblem::tsize        tsize;
                        typedef typename tproblem::tvector      tvector;
                        typedef typename tproblem::tstate       tstate;         ///< optimization state

                        typedef typename tproblem::tulog        tulog;

                        ///
                        /// \brief constructor
                        ///
                        stoch_params(
                                tsize epochs,
                                tsize epoch_size,
                                tscalar alpha0,
                                const tulog& ulog = tulog())
                                :       m_epochs(epochs),
                                        m_epoch_size(epoch_size),
                                        m_alpha0(alpha0),
                                        m_ulog(ulog)
                        {
                        }

                        ///
                        /// \brief destructor
                        ///
                        virtual ~stoch_params()
                        {
                        }

                        ///
                        /// \brief log current optimization state
                        ///
                        void ulog(const tstate& state) const
                        {
                                if (m_ulog) m_ulog(state);
                        }

                        tsize           m_epochs;               ///< number of epochs
                        tsize           m_epoch_size;           ///< epoch size in number of iterations
                        tscalar         m_alpha0;               ///< initial learning rate
                        tulog           m_ulog;                 ///< update log: called after each epoch with the current state
                };
        }
}

