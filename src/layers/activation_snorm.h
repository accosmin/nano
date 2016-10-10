#pragma once

#include "activation.h"

namespace nano
{
        namespace detail
        {
                struct snorm_activation_layer_eval_t
                {
                        template <typename tivector, typename tovector>
                        void operator()(const tivector& idata, tovector&& odata) const
                        {
                                odata = idata.array() / (1 + idata.array().square()).sqrt();
                        }
                };

                struct snorm_activation_layer_grad_t
                {
                        template <typename tgvector, typename tiovector>
                        void operator()(const tgvector& gdata, tiovector&& iodata) const
                        {
                                iodata.array() = gdata.array() *
                                        (1 - iodata.array().square()) *
                                        (1 - iodata.array().square()).sqrt();
                        }
                };
        }

        ///
        /// \brief x/sqrt(1+x^2) activation function
        ///
        struct snorm_activation_layer_t : public activation_layer_t
        <
                detail::snorm_activation_layer_eval_t,
                detail::snorm_activation_layer_grad_t
        >
        {
                NANO_MAKE_CLONABLE(snorm_activation_layer_t)

                // constructor
                explicit snorm_activation_layer_t(const string_t& parameters = string_t()) :
                        activation_layer_t(parameters)
                {
                }
        };
}
