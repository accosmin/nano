#pragma once

#include "decay.hpp"
#include "params.hpp"
#include <limits>

namespace ncv
{
        namespace optim
        {
                ///
                /// \brief common parameters for stochastic optimization
                ///
                template
                <
                        typename tproblem                       ///< optimization problem
                >
                struct stoch_params_t : public params_t<tproblem>
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
                        stoch_params_t( tsize epochs,
                                        tsize epoch_size,
                                        tscalar alpha0,
                                        tscalar decay,
                                        const twlog& wlog = twlog(),
                                        const telog& elog = telog(),
                                        const tulog& ulog = tulog())
                                :       params_t<tproblem>(wlog, elog, ulog),
                                        m_epochs(epochs),
                                        m_epoch_size(epoch_size),
                                        m_alpha0(alpha0),
                                        m_decay(decay),
                                        m_epsilon(std::sqrt(std::numeric_limits<tscalar>::epsilon()))
                        {
                        }

                        ///
                        /// \brief destructor
                        ///
                        virtual ~stoch_params_t()
                        {
                        }

                        ///
                        /// \brief change parameters
                        ///
                        void set_epochs(tsize epochs) { m_epochs = epochs; }
                        void set_epoch_size(tsize epoch_size) { m_epoch_size = epoch_size; }
                        void set_alpha0(tscalar alpha0) { m_alpha0 = alpha0; }
                        void set_decay(tscalar decay) { m_decay = decay; }

                        ///
                        /// \brief current learning rate (following the decay rate)
                        ///
                        tscalar alpha(tsize iter) const { return optim::decay(m_alpha0, iter, m_decay); }

                        ///
                        /// \brief running-average weight
                        ///
                        tscalar weight(tsize k) const
                        {
                                return tscalar(k) / tscalar(m_epochs * m_epoch_size);
                        }

                        tsize           m_epochs;               ///< number of epochs
                        tsize           m_epoch_size;           ///< epoch size in number of iterations
                        tscalar         m_alpha0;               ///< initial learning rate
                        tscalar         m_decay;                ///< learning rate's decay rate
                        tscalar         m_epsilon;              ///< constant
                };
        }
}

