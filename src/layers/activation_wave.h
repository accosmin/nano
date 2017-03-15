#pragma once

#include "activation.h"

namespace nano
{
        ///
        /// \brief exponential wave activation function: x/(exp(-alpha*x)+exp(+alpha*x)).
        ///
        template <int alpha>
        struct ewave_activation_t
        {
                template <typename tiarray>
                static auto output(const tiarray& idata)
                {
                        const auto nexp = (-alpha * idata).exp();
                        const auto pexp = (+alpha * idata).exp();
                        return alpha * idata / (nexp + pexp);
                }

                template <typename tiarray>
                static auto ginput(const tiarray& idata)
                {
                        const auto nexp = (-alpha * idata).exp();
                        const auto pexp = (+alpha * idata).exp();
                        return alpha * (nexp + pexp + alpha * idata * (nexp - pexp)) / (nexp + pexp).square();
                }
        };

        using ewave1_activation_layer_t = activation_layer_t<ewave_activation_t<1>>;
        using ewave2_activation_layer_t = activation_layer_t<ewave_activation_t<2>>;
        using ewave3_activation_layer_t = activation_layer_t<ewave_activation_t<3>>;
        using ewave4_activation_layer_t = activation_layer_t<ewave_activation_t<4>>;
}
