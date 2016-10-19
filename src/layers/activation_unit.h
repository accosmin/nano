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
                        template <typename tivector, typename tovector>
                        static void output(const tivector& idata, tovector&& odata)
                        {
                                odata = idata;
                        }

                        template <typename tgvector, typename tiovector>
                        static void ginput(const tgvector& gdata, tiovector&& iodata)
                        {
                                iodata = gdata;
                        }
                };
        }

        using unit_activation_layer_t = activation_layer_t<detail::unit_activation_t>;
}
