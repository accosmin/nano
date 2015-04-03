#pragma once

#include "gauss.hpp"
#include "random.hpp"
#include "tensor/matrix.hpp"
#include "tensor/transform.hpp"
#include "separable_filter.hpp"

namespace ncv
{
        ///
        /// \brief in-place random additive noise, smoothed with the given Gaussian filter
        ///
        template
        <
                typename tscalar = double,
                typename tmatrix,
                typename tgetter,                       ///< extract value from element (e.g. pixel)
                typename tsetter,                       ///< set value to element (e.g. pixel)

                typename tvalue = typename tmatrix::Scalar
        >
        bool additive_noise(
                const range_t<tscalar>& noise_range,
                const gauss_kernel_t<tscalar>& kernel,
                const range_t<tscalar>& output_range,
                tmatrix& src, tgetter getter, tsetter setter)
        {
                random_t<tscalar> noiser(noise_range.min(), noise_range.max());

                // create random noise map
                typename tensor::matrix_types_t<tscalar>::tmatrix noisemap(src.rows(), src.cols());
                tensor::transform(noisemap, noisemap, [&] (tvalue) { return noiser(); });

                // smooth the noise map
                inplace_separable_filter(kernel, range_t<tscalar>(noiser.min(), noiser.max()), noisemap,
                         [] (tscalar v) { return v; },
                         [] (tscalar, tscalar v) { return v; });

                // add the noise map to the input matrix
                tensor::transform(src, noisemap, src, [&] (tvalue value, tscalar noise)
                {
                        return setter(value, math::cast<tvalue>(output_range.clamp(noise + getter(value))));
                });

                // OK
                return true;
        }
}

