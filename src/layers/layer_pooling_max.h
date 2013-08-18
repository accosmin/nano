#ifndef NANOCV_MAX_POOLING_LAYER_H
#define NANOCV_MAX_POOLING_LAYER_H

#include "layer_pooling.h"
#include "core/numeric.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // max-pooling layer.
        /////////////////////////////////////////////////////////////////////////////////////////

        class max_pooling_layer_t : public pooling_layer_t
        {
        public:

                // constructor
                max_pooling_layer_t(const string_t& = string_t()) {}

                NCV_MAKE_CLONABLE(max_pooling_layer_t, layer_t, "max pooling layer")

        protected:

                // pool outputs & gradients
                virtual scalar_t forward_pool(scalar_t ox, scalar_t ix) const
                {
                        return std::max(ix, ox);
                }
                virtual scalar_t backward_pool(scalar_t gx, scalar_t ox, scalar_t ix) const
                {
                        const bool cond = math::equal(ox, ix);
                        return gx * math::kronocker<scalar_t>(cond);
                }
        };
}

#endif // NANOCV_MAX_POOLING_LAYER_H
