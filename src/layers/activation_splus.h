#pragma once

#include "activation.h"

namespace nano
{
        ///
        /// \brief soft-plus log(1 + exp(x)) activation function.
        ///
        struct softplus_activation_t
        {
                template <typename tiarray>
                static auto output(const tiarray& idata)
                {
                        return (1 + idata.exp()).log();
                }

                template <typename tiarray>
                static auto ginput(const tiarray& idata)
                {
                        return idata.exp() / (1 + idata.exp());
                }
        };

        using softplus_activation_layer_t = activation_layer_t<softplus_activation_t>;
}
