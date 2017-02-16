#pragma once

#include "activation.h"

namespace nano
{
        ///
        /// \brief sin(x) activation function.
        ///
        namespace detail
        {
                struct sin_activation_t
                {
                        template <typename tivector>
                        static auto output(const tivector& idata)
                        {
                                return idata.array().sin();
                        }

                        template <typename tivector, typename tovector>
                        static auto ginput(const tivector& idata, const tovector&)
                        {
                                return idata.array().cos();
                        }
                };
        }

        using sin_activation_layer_t = activation_layer_t<detail::sin_activation_t>;
}
