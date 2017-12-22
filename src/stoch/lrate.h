#pragma once

#include <cmath>
#include <cassert>
#include "scalar.h"

namespace nano
{
        ///
        /// \brief learning rate as a function of:
        ///     - alpha0        - the initial learning rate
        ///     - decay         - the decay rate factor
        ///     - iter          - current iteration
        ///     - epoch_size    - epoch size in number of iterators
        ///
        /// learning rate = alpha0 / (1 + iter/epoch_size)^decay
        ///
        class lrate_t
        {
        public:
                ///
                /// \brief constructor
                ///
                lrate_t(const scalar_t alpha0, const scalar_t decay, const size_t epoch_size) :
                        m_alpha0(alpha0),
                        m_decay(decay),
                        m_epoch_size(epoch_size)
                {
                        assert(decay >= scalar_t(0));
                        assert(decay <= scalar_t(1));
                        assert(alpha0 > scalar_t(0));
                        assert(m_epoch_size > 0);
                }

                ///
                /// \brief update the current learning rate for the given iteration
                ///
                scalar_t get()
                {
                        const auto base = 1 + static_cast<scalar_t>(m_iter ++) / static_cast<scalar_t>(m_epoch_size);
                        return m_alpha0 / std::pow(base, m_decay);
                }

        private:

                // attributes
                scalar_t        m_alpha0;               ///<
                scalar_t        m_decay;                ///<
		size_t		m_iter{0};              ///< current iteration
                size_t          m_epoch_size{0};        ///<
        };
}
