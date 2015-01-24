#pragma once

#include "random.hpp"
#include "gaussian.hpp"
#include "tensor/matrix.hpp"
#include "tensor/transform.hpp"

namespace ncv
{
        ///
        /// \brief in-place random additive noise [offset - range, offset + range]
        ///
        /// \note the noise is smoothed with a Gaussian filter having the given sigma
        ///
        template
        <
                typename tscalar = double,
                typename tmatrix,
                typename tgetter,                       ///< extract value from element (e.g. pixel)
                typename tsetter,                       ///< set value to element (e.g. pixel)

                typename tvalue = typename tmatrix::Scalar
        >
        bool additive_noise(tmatrix& src, tscalar offset, tscalar range, tscalar sigma, tscalar cutoff,
                tscalar minv, tscalar maxv, tgetter getter, tsetter setter)
        {
                random_t<tscalar> noiser(offset - range, offset + range);

                // create random noise map
                typename tensor::matrix_types_t<tscalar>::tmatrix noisemap(src.rows(), src.cols());
                tensor::transform(noisemap, noisemap, [&] (tvalue) { return noiser(); });

                // smooth the noise map
                gaussian(noisemap, sigma, cutoff, noiser.min(), noiser.max(),
                         [] (tscalar v) { return v; },
                         [] (tscalar, tscalar v) { return v; });

                // add the noise map to the input matrix
                tensor::transform(src, noisemap, src, [&] (tvalue v, tscalar n)
                {
                        return setter(v, math::cast<tvalue>(math::clamp(n + getter(v), minv, maxv)));
                });

                // OK
                return true;
        }
}

