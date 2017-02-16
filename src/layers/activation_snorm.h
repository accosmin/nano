#pragma once

#include "activation.h"

namespace nano
{
        ///
        /// \brief x/sqrt(1+x^2) activation function.
        ///
        namespace detail
        {
                struct snorm_activation_t
                {
                        template <typename tivector>
                        static auto output(const tivector& idata)
                        {
                                return idata.array() / (1 + idata.array().square()).sqrt();
                        }

                        template <typename tivector, typename tovector>
                        static auto ginput(const tivector&, const tovector& odata)
                        {
                                return (1 - odata.array().square()) * (1 - odata.array().square()).sqrt();
                        }
                };
        }

        using snorm_activation_layer_t = activation_layer_t<detail::snorm_activation_t>;
}
