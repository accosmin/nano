#pragma once

#include "activation.h"

namespace nano
{
        namespace detail
        {
                struct unit_activation_layer_eval_t
                {
                        template <typename tivector, typename tovector>
                        void operator()(const tivector& idata, tovector&& odata) const
                        {
                                odata = idata;
                        }
                };

                struct unit_activation_layer_grad_t
                {
                        template <typename tgvector, typename tiovector>
                        void operator()(const tgvector& gdata, tiovector&& iodata) const
                        {
                                iodata = gdata;
                        }
                };
        }

        ///
        /// \brief identity activation function
        ///
        struct unit_activation_layer_t : public activation_layer_t
        <
                detail::unit_activation_layer_eval_t,
                detail::unit_activation_layer_grad_t
        >
        {
                NANO_MAKE_CLONABLE(unit_activation_layer_t, "identity activation layer", "")

                // constructor
                explicit unit_activation_layer_t(const string_t& parameters = string_t()) :
                        activation_layer_t(parameters)
                {
                }
        };
}
