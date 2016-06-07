#pragma once

#include "activation.h"

namespace nano
{
        namespace detail
        {
                struct tanh_activation_layer_eval_t
                {
                        template <typename tivector, typename tovector>
                        void operator()(const tivector& idata, tovector&& odata) const
                        {
                                odata.array() =
                                        (idata.array().exp() - (-idata.array()).exp()) /
                                        (idata.array().exp() + (-idata.array()).exp());
                        }
                };

                struct tanh_activation_layer_grad_t
                {
                        template <typename tgvector, typename tiovector>
                        void operator()(const tgvector& gdata, tiovector&& iodata) const
                        {
                                iodata.array() = gdata.array() * (1 - iodata.array() * iodata.array());
                        }
                };
        }

        ///
        /// \brief hyperbolic tangent activation function
        ///
        class tanh_activation_layer_t : public activation_layer_t
        <
                detail::tanh_activation_layer_eval_t,
                detail::tanh_activation_layer_grad_t
        >
        {
        public:

                NANO_MAKE_CLONABLE(tanh_activation_layer_t, "hyperbolic tangent activation layer")

                // constructor
                explicit tanh_activation_layer_t(const string_t& parameters = string_t()) :
                        activation_layer_t(parameters)
                {
                }
        };
}
