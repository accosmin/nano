#pragma once

#include <cmath>
#include <cassert>
#include "scalar.h"

namespace nano
{
        ///
        /// \brief compute the current learning rate as a function of:
        ///     - alpha0        - the initial learning rate
        ///     - iter          - iteration
        ///     - decay         - the decay rate mode
        ///
        /// learning rate = alpha0 / (iter + 1)^decay
        ///
        struct lrate_t
        {
                ///
                /// \brief constructor
                ///
                lrate_t(const scalar_t alpha0, const scalar_t decay) :
                        m_alpha0(alpha0),
                        m_decay(decay),
                        m_iteration(0)
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
                        return m_alpha0 / std::pow(++ m_iteration, m_decay);
                }

                // attributes
                scalar_t        m_alpha0;
                scalar_t        m_decay;
                scalar_t        m_iteration;
        };

}

