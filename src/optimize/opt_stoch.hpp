#pragma once

#include <cmath>

namespace ncv
{
        namespace optimize
        {
                ///
                /// \brief learning rate's decay rate as a function of the iteration
                ///
                enum class decay_rate : int
                {
                        unit,           ///< 1/(iteration)^1.00
                        qrt3,           ///< 1/(iteration)^0.75
                        sqrt            ///< 1/(iteration)^0.50
                };

                ///
                /// \brief compute the current learning rate as a function of:
                ///     - alpha0        - the initial learning rate
                ///     - iter          - the current iteration
                ///     - decay_rate    - the decay rate mode
                ///
                template
                <
                        typename tscalar,
                        typename tsize
                >
                tscalar decay(tscalar alpha0, tsize iter, decay_rate mode)
                {
                        const tscalar expo = static_cast<tscalar>(iter + 1);

                        switch (mode)
                        {
                        case decay_rate::unit:  return static_cast<tscalar>(alpha0 / std::pow(expo, tscalar(1.00)));
                        case decay_rate::qrt3:  return static_cast<tscalar>(alpha0 / std::pow(expo, tscalar(0.75)));
                        case decay_rate::sqrt:
                        default:                return static_cast<tscalar>(alpha0 / std::pow(expo, tscalar(0.50)));
                        }
                }
        }
}

