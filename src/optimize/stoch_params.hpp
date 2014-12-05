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

                        typedef typename tproblem::twlog        twlog;
                        typedef typename tproblem::telog        telog;
                        typedef typename tproblem::tulog        tulog;

                        ///
                        /// \brief constructor
                        ///
                        stoch_params(   tsize epochs,
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
                        /// \brief change logger
                        ///
                        void set_ulog(const tulog& ulog) { m_ulog = ulog; }

                        ///
                        /// \brief change parameters
                        ///
                        void set_epochs(size_t epochs) { m_epochs = epochs; }
                        void set_epoch_size(size_t epoch_size) { m_epoch_size = epoch_size; }
                        void set_alpha0(size_t alpha0) { m_alpha0 = alpha0; }

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

