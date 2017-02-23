#pragma once

#include "activation.h"

namespace nano
{
        ///
        /// \brief hyperbolic tangent activation function.
        ///
        struct tanh_activation_t
        {
                template <typename tivector>
                static auto output(const tivector& idata)
                {
                        return idata.array().tanh();
                }

                template <typename tivector, typename tovector>
                static auto ginput(const tivector&, const tovector& odata)
                {
                        return (1 - odata.array().square());
                }
        };

        using tanh_activation_layer_t = activation_layer_t<tanh_activation_t>;
}
