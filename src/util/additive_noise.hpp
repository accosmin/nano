#pragma once

#include "cast.hpp"
#include "math.hpp"
#include "random.hpp"
#include <algorithm>

namespace ncv
{
        ///
        /// \brief add in-place random noise with a given offset and a dynamic range
        ///
        /// the noise map is filtered with a Gaussian kernel having the given standard deviation sigma
        ///
        template
        <
                typename tscalar = double,
                typename tmatrix,
                typename tgetter,                       ///< extract value from element (e.g. pixel)
                typename tsetter,                       ///< set value to element (e.g. pixel)

                typename tvalue = typename tmatrix::Scalar
        >
        bool additive_noise(tmatrix& src, tscalar offset, tscalar range, tscalar sigma,
                tvalue minv, tvalue maxv, tgetter getter, tsetter setter)
        {
                random_t<tscalar> noiser(offset - range, offset + range);

                // TODO: store the noise map into a matrix & smoothed with the given sigma!

                std::transform(src.data(), src.data() + src.size(), src.data(), [&] (tvalue v)
                {
                        return setter(v, math::cast<tvalue>(math::clamp(noiser() + getter(v), minv, maxv)));
                });

                return true;
        }
}

