#pragma once

#include "activation.h"

namespace nano
{
        ///
        /// \brief soft-plus (max approximation) activation function.
        ///
        namespace detail
        {
                struct softplus_activation_t
                {
                        template <typename tivector, typename tovector>
                        static void output(const tivector& idata, tovector&& odata)
                        {
                                odata.array() = (1 + idata.array().exp()).log();
                        }

                        template <typename tgvector, typename tiovector>
                        static void ginput(const tgvector& gdata, tiovector&& iodata)
                        {
                                iodata.array() = gdata.array() * (1 - (- iodata.array()).exp());
                        }
                };
        }

        using softplus_activation_layer_t = activation_layer_t<detail::softplus_activation_t>;
}
