#pragma once

#include <cmath>
#include <cassert>
#include "scalar.h"

namespace nano
{
        ///
        /// \brief learning rate as an exponential decay function of:
        ///     - alpha0        - the initial learning rate
        ///     - decay         - the decay rate mode
        ///     - iter          - current iteration
	///	- epoch_size	- epoch size in iterations
        ///
        /// learning rate = alpha0 * decay^{iter/epoch_size}
        ///
        struct lrate_t
        {
                ///
                /// \brief constructor
                ///
                lrate_t(const scalar_t alpha0, const scalar_t decay, const size_t epoch_size) :
                        m_alpha0(alpha0),
                        m_decay(decay),
			m_epoch_size(epoch_size),
                        m_iteration(0)
                {
                        assert(decay > scalar_t(0));
                        assert(decay <= scalar_t(1));
                        assert(alpha0 > scalar_t(0));
                }

                ///
                /// \brief update the current learning rate for the given iteration
                ///
                scalar_t get()
                {
                        const auto power = static_cast<scalar_t>(m_iteration ++) / static_cast<scalar_t>(m_epoch_size);
                        return m_alpha0 * std::pow(m_decay, power);
                }

                // attributes
                scalar_t        m_alpha0;
                scalar_t        m_decay;
		size_t		m_epoch_size;
		size_t		m_iteration;
        };

}

