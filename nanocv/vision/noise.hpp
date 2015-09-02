#pragma once

#include "nanocv/math/gauss.hpp"
#include "nanocv/math/random.hpp"
#include "nanocv/tensor/matrix.hpp"
#include "nanocv/tensor/transform.hpp"

namespace ncv
{
        ///
        /// \brief in-place random additive noise, smoothed with the given Gaussian filter
        ///
        template
        <
                typename tmatrix,
                typename tscalar,
                typename trange,
                typename tvalue = typename tmatrix::Scalar
        >
        void additive_noise(
                tmatrix&& srcplane,
                const trange noise_range,               ///< noise range
                const tscalar sigma,                    ///< Gaussian sigma
                const trange output_range)
        {
                random_t<tscalar> noiser(noise_range.min(), noise_range.max());

                // create random noise map
                typename tensor::matrix_t<tscalar> noisemap(srcplane.rows(), srcplane.cols());
                tensor::transform(noisemap, noisemap, [&] (tvalue) { return noiser(); });

                // smooth the noise map
                separable_filter(gauss_kernel_t<tscalar>(sigma).get(), noisemap);

                // add the noise map to the input matrix
                tensor::transform(srcplane, noisemap, srcplane, [&] (tvalue value, tscalar noise)
                {
                        return math::cast<tvalue>(output_range.clamp(value + noise));
                });
        }
}

