#pragma once

#include <cmath>
#include <cassert>

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
        template
        <
                typename tscalar
        >
        struct lrate_t
        {
                ///
                /// \brief constructor
                ///
                lrate_t(const tscalar alpha0, const tscalar decay)
                        :       m_alpha0(alpha0),
                                m_decay(decay)
                {
                        assert(decay >= tscalar(0));
                        assert(decay <= tscalar(1));
                        assert(alpha0 > tscalar(0));
                }

                ///
                /// \brief update the current learning rate for the given iteration
                ///
                template
                <
                        typename tsize
                >
                tscalar get(const tsize iter) const
                {
                        const tscalar base = static_cast<tscalar>(iter + 1);

                        return m_alpha0 / static_cast<tscalar>(std::pow(base, m_decay));
                }

                // attributes
                tscalar         m_alpha0;
                tscalar         m_decay;
        };

}

