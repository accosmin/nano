#pragma once

#include "activation.h"

namespace nano
{
        ///
        /// \brief x/(1+x^2) activation function.
        ///
        struct wave1_activation_t
        {
                template <typename tivector>
                static auto output(const tivector& idata)
                {
                        return idata.array() / (1 + idata.array().square());
                }

                template <typename tivector, typename tovector>
                static auto ginput(const tivector& idata, const tovector&)
                {
                        return (1 - idata.array().square()) / (1 + idata.array().square()).square();
                }
        };

        ///
        /// \brief x/(1+x^4) activation function.
        ///
        struct wave2_activation_t
        {
                template <typename tivector>
                static auto output(const tivector& idata)
                {
                        return idata.array() / (1 + idata.array().square().square());
                }

                template <typename tivector, typename tovector>
                static auto ginput(const tivector& idata, const tovector&)
                {
                        return (1 - 3 * idata.array().square().square()) / (1 + idata.array().square().square()).square();
                }
        };

        using wave1_activation_layer_t = activation_layer_t<wave1_activation_t>;
        using wave2_activation_layer_t = activation_layer_t<wave2_activation_t>;
}
