#pragma once

#include <cmath>
#include <cassert>

namespace ncv
{
        namespace min
        {
                ///
                /// \brief compute the current learning rate as a function of:
                ///     - alpha0        - the initial learning rate
                ///     - iter          - the current iteration
                ///     - power         - the decay rate mode
                ///
                /// learning rate = alpha0 / (iter + 1)^rate
                ///
                template
                <
                        typename tscalar,
                        typename tsize
                >
                tscalar decay(tscalar alpha0, tsize iter, tscalar rate)
                {
                        assert(rate >= tscalar(0));
                        assert(rate <= tscalar(1));
                        assert(alpha0 > tscalar(0));

                        const tscalar base = static_cast<tscalar>(iter + 1);

                        return alpha0 / static_cast<tscalar>(std::pow(base, rate));
                }
        }
}

