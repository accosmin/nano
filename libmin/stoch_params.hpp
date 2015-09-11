#pragma once

#include "stoch_decay.hpp"
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
                                        const twlog& w = twlog(),
                                        const telog& e = telog(),
                                        const tulog& u = tulog())
                                :       params_t<tproblem>(w, e, u),
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
                        /// \brief current learning rate (following the decay rate)
                        ///
                        tscalar alpha(tsize iter) const
                        {
                                return optim::decay(m_alpha0, iter, m_decay);
                        }

                        ///
                        /// \brief running-average weight
                        ///
                        tscalar weight(tsize k) const
                        {
                                return tscalar(k) / tscalar(m_epochs * m_epoch_size);
                        }

                        // attributes
                        tsize           m_epochs;               ///< number of epochs
                        tsize           m_epoch_size;           ///< epoch size in number of iterations
                        tscalar         m_alpha0;               ///< initial learning rate
                        tscalar         m_decay;                ///< learning rate's decay rate
                        tscalar         m_epsilon;              ///< constant
                };
        }
}

