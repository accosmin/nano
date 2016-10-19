#pragma once

#include "activation.h"

namespace nano
{
        ///
        /// \brief hyperbolic tangent activation function.
        ///
        namespace detail
        {
                struct tanh_activation_t
                {
                        template <typename tivector, typename tovector>
                        static void output(const tivector& idata, tovector&& odata)
                        {
                                odata.array() = idata.array().tanh();
                        }

                        template <typename tgvector, typename tiovector>
                        static void ginput(const tgvector& gdata, tiovector&& iodata)
                        {
                                iodata.array() = gdata.array() * (1 - iodata.array().square());
                        }
                };
        }

        using tanh_activation_layer_t = activation_layer_t<detail::tanh_activation_t>;
}
