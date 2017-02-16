#pragma once

#include "activation.h"

namespace nano
{
        ///
        /// \brief x/(1+x^2) activation function.
        ///
        namespace detail
        {
                struct swave_activation_t
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
        }

        using swave_activation_layer_t = activation_layer_t<detail::swave_activation_t>;
}
