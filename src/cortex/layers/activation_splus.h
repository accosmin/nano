#pragma once

#include "activation.h"

namespace nano
{
        namespace detail
        {
                struct softplus_activation_layer_eval_t
                {
                        template <typename tivector, typename tovector>
                        void operator()(const tivector& idata, tovector&& odata) const
                        {
                                odata.array() = (1 + idata.array().exp()).log();
                        }
                };

                struct softplus_activation_layer_grad_t
                {
                        template <typename tgvector, typename tiovector>
                        void operator()(const tgvector& gdata, tiovector&& iodata) const
                        {
                                iodata.array() = gdata.array() * (1 - (- iodata.array()).exp());
                        }
                };
        }

        ///
        /// \brief soft-plus (max approximation) activation function
        ///
        class softplus_activation_layer_t : public activation_layer_t
        <
                detail::softplus_activation_layer_eval_t,
                detail::softplus_activation_layer_grad_t
        >
        {
        public:

                NANO_MAKE_CLONABLE(softplus_activation_layer_t, "soft-plus activation layer")

                // constructor
                explicit softplus_activation_layer_t(const string_t& parameters = string_t()) :
                        activation_layer_t(parameters)
                {
                }
        };
}
