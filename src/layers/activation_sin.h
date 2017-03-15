#pragma once

#include "activation.h"

namespace nano
{
        ///
        /// \brief sin(x) activation function.
        ///
        struct sin_activation_t
        {
                template <typename tiarray>
                static auto output(const tiarray& idata)
                {
                        return idata.sin();
                }

                template <typename tiarray>
                static auto ginput(const tiarray& idata)
                {
                        return idata.cos();
                }
        };

        using sin_activation_layer_t = activation_layer_t<sin_activation_t>;
}
