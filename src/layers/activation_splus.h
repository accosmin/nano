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
        }

        using softplus_activation_layer_t = activation_layer_t<detail::softplus_activation_t>;
}
