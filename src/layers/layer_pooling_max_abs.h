#ifndef NANOCV_MAX_ABS_POOLING_LAYER_H
#define NANOCV_MAX_ABS_POOLING_LAYER_H

#include "layer_pooling.h"
#include "core/numeric.hpp"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // max-absolute-pooling layer.
        /////////////////////////////////////////////////////////////////////////////////////////

        class max_abs_pooling_layer_t : public pooling_layer_t
        {
        public:

                // constructor
                max_abs_pooling_layer_t(const string_t& = string_t()) {}

                NCV_MAKE_CLONABLE(max_abs_pooling_layer_t, layer_t, "max absolute pooling layer")

        protected:

                // pool outputs & gradients
                virtual scalar_t forward_pool(scalar_t ox, scalar_t ix) const
                {
                        return std::fabs(ix) > std::fabs(ox) ? ix : ox;
                }
                virtual scalar_t backward_pool(scalar_t gx, scalar_t ox, scalar_t ix) const
                {
                        const bool cond = math::equal(std::fabs(ox), std::fabs(ix));
                        return gx * math::kronocker<scalar_t>(cond);
                }
        };
}

#endif // NANOCV_MAX_ABS_POOLING_LAYER_H
