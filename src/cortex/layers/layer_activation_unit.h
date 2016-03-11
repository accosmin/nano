#pragma once

#include "layer_activation.h"

namespace nano
{
        namespace detail
        {
                struct unit_activation_layer_eval_t
                {
                        scalar_t operator()(scalar_t x) const
                        {
                                return x;
                        }
                };

                struct unit_activation_layer_grad_t
                {
                        scalar_t operator()(scalar_t g, scalar_t o) const
                        {
                                NANO_UNUSED1(o);

                                return g;
                        }
                };
        }

        ///
        /// \brief identity activation function
        ///
        class unit_activation_layer_t : public activation_layer_t
        <
                detail::unit_activation_layer_eval_t,
                detail::unit_activation_layer_grad_t
        >
        {
        public:

                NANO_MAKE_CLONABLE(unit_activation_layer_t, "identity activation layer")

                // constructor
                explicit unit_activation_layer_t(const string_t& parameters = string_t())
                        :       activation_layer_t(parameters)
                {
                }
        };
}
