#pragma once

#include <cmath>
#include <cassert>
#include "scalar.h"

namespace nano
{
        ///
        /// \brief learning rate as an exponential decay function of:
        ///     - alpha0        - the initial learning rate
        ///     - decay         - the decay rate factor
        ///     - iter          - current iteration
        ///
        /// learning rate = alpha0 * (1 + iteration)^decay
        ///
        /// see "Online Learning and Stochastic Approximations", by Leon Bottou
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
                        return m_alpha0 / std::pow(static_cast<scalar_t>(++ m_iteration), m_decay);
                }

                // attributes
                scalar_t        m_alpha0;
                scalar_t        m_decay;
		size_t		m_iteration;
        };

}

