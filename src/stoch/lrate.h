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
        ///     - epoch_size    - number of iterations per epoch
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
                        m_inv_epoch_size(scalar_t(1) / static_cast<scalar_t>(epoch_size))
                {
                        assert(decay >= scalar_t(0));
                        assert(decay <= scalar_t(1));
                        assert(alpha0 > scalar_t(0));
                }

                ///
                /// \brief update the current learning rate for the given iteration
                ///
                scalar_t get()
                {
                        const auto base = 1 + static_cast<scalar_t>(m_iter ++) * m_inv_epoch_size;
                        return m_alpha0 / std::pow(base, m_decay);
                }

        private:

                // attributes
                scalar_t        m_alpha0;               ///<
                scalar_t        m_decay;                ///<
                scalar_t        m_inv_epoch_size;       ///< 1/epoch_size
	        size_t		m_iter{0};              ///< current iteration
        };
}
