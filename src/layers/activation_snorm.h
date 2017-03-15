#pragma once

#include "activation.h"

namespace nano
{
        ///
        /// \brief x/sqrt(1+x^2) activation function.
        ///
        struct snorm_activation_t
        {
                template <typename tiarray>
                static auto output(const tiarray& idata)
                {
                        return idata / (1 + idata.square()).sqrt();
                }

                template <typename tiarray>
                static auto ginput(const tiarray& idata)
                {
                        return 1 / (1 + idata.square()).cube().sqrt();
                }
        };

        using snorm_activation_layer_t = activation_layer_t<snorm_activation_t>;
}
