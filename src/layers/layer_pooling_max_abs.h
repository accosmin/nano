#ifndef NANOCV_MAX_ABS_POOLING_LAYER_H
#define NANOCV_MAX_ABS_POOLING_LAYER_H

#include "layer_pooling.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // max-absolute-pooling layer.
        /////////////////////////////////////////////////////////////////////////////////////////

        namespace impl
        {
                struct max_abs_pooling_layer_wgrad_op
                {
                        void operator()(scalar_t x, scalar_t& w, scalar_t& g) const
                        {
                                static const scalar_t beta = 1.0;
                                const scalar_t e = exp(beta * x), ie = 1.0 / e;
                                w = e + ie;
                                g = beta * (e - ie);
                        }
                };
        }

        class max_abs_pooling_layer_t : public pooling_layer_t<impl::max_abs_pooling_layer_wgrad_op>
        {
        public:

                // constructor
                max_abs_pooling_layer_t(const string_t& = string_t()) {}

                NCV_MAKE_CLONABLE(max_abs_pooling_layer_t, layer_t, "max absolute pooling layer")
        };
}

#endif // NANOCV_MAX_ABS_POOLING_LAYER_H
