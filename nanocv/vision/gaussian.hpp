#pragma once

#include "separable_filter.hpp"
#include "nanocv/math/gauss.hpp"

namespace ncv
{
        ///
        /// \brief filter the input tensor using a Gaussian filter with the given sigma
        ///
        template
        <
                typename ttensor,
                typename tscalar
        >
        ttensor gaussian(const ttensor& src, const tscalar sigma)
        {
                const gauss_kernel_t<tscalar> kernel(sigma);

                ttensor dst = src;

                for (int _od = 0; _od < src.dims(); _od ++)
                {
                        inplace_separable_filter(kernel, dst.matrix(_od));
                }

                return dst;
        }
}

