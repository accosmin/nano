#ifndef NANOCV_MAX_POOLING_LAYER_H
#define NANOCV_MAX_POOLING_LAYER_H

#include "layer_pooling.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // max-pooling layer.
        /////////////////////////////////////////////////////////////////////////////////////////

        namespace impl
        {
                struct max_pooling_layer_wgrad_op
                {
                        void operator()(scalar_t x, scalar_t& w, scalar_t& g) const
                        {
                                static const scalar_t beta = 1.0;
                                w = exp(beta * x);
                                g = beta * w;
                        }
                };
        }

        class max_pooling_layer_t : public pooling_layer_t<impl::max_pooling_layer_wgrad_op>
        {
        public:

                // constructor
                max_pooling_layer_t(const string_t& = string_t()) {}

                NCV_MAKE_CLONABLE(max_pooling_layer_t, layer_t, "max pooling layer")
        };
}

#endif // NANOCV_MAX_POOLING_LAYER_H
