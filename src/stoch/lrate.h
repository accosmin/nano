#pragma once

#include <cassert>
#include "scalar.h"

namespace nano
{
        ///
        /// \brief learning rate as a polynomial function of:
        ///     - alpha0        - the initial learning rate
        ///     - decay         - the decay rate factor
        ///     - iter          - current iteration
        ///     - size          - batch size (e.g. #samples in the sum)
        ///
        /// learning rate = alpha0 / (1 + iter/size)^decay
        ///
        struct lrate_t
        {
                ///
                /// \brief constructor
                ///
                lrate_t(const scalar_t alpha0, const scalar_t decay, const size_t size) :
                        m_alpha0(alpha0),
                        m_decay(decay),
                        m_iter(0),
                        m_size(size)
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
                        const auto base = scalar_t(1) + static_cast<scalar_t>(m_iter ++) / static_cast<scalar_t>(m_size);
                        return m_alpha0 / std::pow(base, m_decay);
                }

                // attributes
                scalar_t        m_alpha0;       ///<
                scalar_t        m_decay;        ///<
		size_t		m_iter;         ///< current iteration
                size_t          m_size;         ///< #epoch size
        };

}

