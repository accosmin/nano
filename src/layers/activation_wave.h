#pragma once

#include "activation.h"

namespace nano
{
        ///
        /// \brief polynomial wave activation function: x/(1+x^degree).
        ///
        template <int degree>
        struct pwave_activation_t
        {
                template <typename tivector>
                static auto output(const tivector& idata)
                {
                        const auto ipow = idata.array().pow(degree);
                        return idata.array() / (1 + ipow);
                }

                template <typename tivector, typename tovector>
                static auto ginput(const tivector& idata, const tovector&)
                {
                        const auto ipow = idata.array().pow(degree);
                        return (1 - (degree - 1) * ipow) / (1 + ipow).square();
                }
        };

        ///
        /// \brief exponential wave activation function: x/(exp(-alpha*x)+exp(+alpha*x)).
        ///
        template <int alpha>
        struct ewave_activation_t
        {
                template <typename tivector>
                static auto output(const tivector& idata)
                {
                        const auto nexp = (-alpha * idata.array()).exp();
                        const auto pexp = (+alpha * idata.array()).exp();
                        return alpha * idata.array() / (nexp + pexp);
                }

                template <typename tivector, typename tovector>
                static auto ginput(const tivector& idata, const tovector&)
                {
                        const auto nexp = (-alpha * idata.array()).exp();
                        const auto pexp = (+alpha * idata.array()).exp();
                        return alpha * (nexp + pexp + alpha * idata.array() * (nexp - pexp)) / (nexp + pexp).square();
                }
        };

        using pwave2_activation_layer_t = activation_layer_t<pwave_activation_t<2>>;
        using pwave4_activation_layer_t = activation_layer_t<pwave_activation_t<4>>;

        using ewave1_activation_layer_t = activation_layer_t<ewave_activation_t<1>>;
        using ewave2_activation_layer_t = activation_layer_t<ewave_activation_t<2>>;
        using ewave3_activation_layer_t = activation_layer_t<ewave_activation_t<3>>;
        using ewave4_activation_layer_t = activation_layer_t<ewave_activation_t<4>>;
}
