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
                        template <typename tivector, typename tovector>
                        static void output(const tivector& idata, tovector&& odata)
                        {
                                odata = idata.array() / (1 + idata.array().square()).sqrt();
                        }

                        template <typename tgvector, typename tiovector>
                        static void ginput(const tgvector& gdata, tiovector&& iodata)
                        {
                                iodata.array() = gdata.array() *
                                        (1 - iodata.array().square()) *
                                        (1 - iodata.array().square()).sqrt();
                        }
                };
        }

        using snorm_activation_layer_t = activation_layer_t<detail::snorm_activation_t>;
}
