#ifndef NANOCV_UNIT_ACTIVATION_LAYER_H
#define NANOCV_UNIT_ACTIVATION_LAYER_H

#include "layer_activation.h"

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////
        // identity activation function.
        ////////////////////////////////////////////////////////////////////////////////
	
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

        class unit_activation_layer_t : public activation_layer_t
        <
                impl::unit_activation_layer_eval_t,
                impl::unit_activation_layer_grad_t
        >
        {
        public:

                // constructor
                unit_activation_layer_t(const string_t& = string_t()) {}

                NCV_MAKE_CLONABLE(unit_activation_layer_t, layer_t, "identity activation layer")
        };
}

#endif // NANOCV_UNIT_ACTIVATION_LAYER_H
