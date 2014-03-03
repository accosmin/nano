#ifndef NANOCV_UNIT_ACTIVATION_LAYER_H
#define NANOCV_UNIT_ACTIVATION_LAYER_H

#include "layer_activation.h"

namespace ncv
{
        namespace impl
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
                                return g;
                        }
                };
        }

        ///
        /// identity activation function
        ///
        class unit_activation_layer_t : public activation_layer_t
        <
                impl::unit_activation_layer_eval_t,
                impl::unit_activation_layer_grad_t
        >
        {
        public:

                // constructor
                unit_activation_layer_t(const string_t& parameters = string_t())
                        :       activation_layer_t(parameters, "identity activation layer")
                {
                }

                // create an object clone
                virtual rlayer_t clone(const string_t& parameters) const
                {
                        return rlayer_t(new unit_activation_layer_t(parameters));
                }
        };
}

#endif // NANOCV_UNIT_ACTIVATION_LAYER_H
