#pragma once

#include "activation.h"

namespace nano
{
        ///
        /// \brief soft-plus log(1 + exp(x)) activation function.
        ///
        struct softplus_activation_t
        {
                template <typename tivector>
                static auto output(const tivector& idata)
                {
                        return (1 + idata.array().exp()).log();
                }

                template <typename tivector, typename tovector>
                static auto ginput(const tivector&, const tovector& odata)
                {
                        return (1 - (-odata.array()).exp());
                }
        };

        using softplus_activation_layer_t = activation_layer_t<softplus_activation_t>;
}
