#pragma once

#include "activation.h"

namespace nano
{
        ///
        /// \brief identity activation function.
        ///
        namespace detail
        {
                struct unit_activation_t
                {
                        template <typename tivector>
                        static auto output(const tivector& idata)
                        {
                                return idata.array();
                        }

                        template <typename tivector, typename tovector>
                        static auto ginput(const tivector&, const tovector&)
                        {
                                return 1;
                        }
                };
        }

        using unit_activation_layer_t = activation_layer_t<detail::unit_activation_t>;
}
