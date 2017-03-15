#pragma once

#include "activation.h"

namespace nano
{
        ///
        /// \brief hyperbolic tangent activation function.
        ///
        struct tanh_activation_t
        {
                template <typename tiarray>
                static auto output(const tiarray& idata)
                {
                        return idata.tanh();
                }

                template <typename tiarray>
                static auto ginput(const tiarray& idata)
                {
                        return 4 / (idata.exp() + (-idata).exp()).square();
                }
        };

        ///
        /// \brief sigmoid activation function.
        ///
        struct sigm_activation_t
        {
                template <typename tiarray>
                static auto output(const tiarray& idata)
                {
                        return idata.exp() / (1 + idata.exp());
                }

                template <typename tiarray>
                static auto ginput(const tiarray& idata)
                {
                        return idata.exp() / (1 + idata.exp()).square();
                }
        };

        using tanh_activation_layer_t = activation_layer_t<tanh_activation_t>;
        using sigm_activation_layer_t = activation_layer_t<sigm_activation_t>;
}
