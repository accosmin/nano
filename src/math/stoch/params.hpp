#pragma once

#include "decay.hpp"
#include "math/params.hpp"
#include <limits>

namespace math
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
                using param_t = stoch_params_t<tproblem>;
                using tstate = typename param_t::tstate;
                using tscalar = typename param_t::tscalar;
                using tvector = typename param_t::tvector;
                using topulog = typename param_t::topulog;

                ///
                /// \brief constructor
                ///
                stoch_params_t( std::size_t epochs,
                                std::size_t epoch_size,
                                tscalar alpha0,
                                tscalar decay,
                                const topulog& u = topulog())
                        :       params_t<tproblem>(u),
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
                tscalar alpha(std::size_t iter) const
                {
                        return math::decay(m_alpha0, iter, m_decay);
                }

                ///
                /// \brief running-average weight
                ///
                tscalar weight(std::size_t k) const
                {
                        return static_cast<tscalar>(k) / static_cast<tscalar>(m_epochs * m_epoch_size);
                }

                // attributes
                std::size_t     m_epochs;               ///< number of epochs
                std::size_t     m_epoch_size;           ///< epoch size in number of iterations
                tscalar         m_alpha0;               ///< initial learning rate
                tscalar         m_decay;                ///< learning rate's decay rate
                tscalar         m_epsilon;              ///< constant
        };
}
