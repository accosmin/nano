#pragma once

#include <cmath>
#include "tensor.h"

namespace nano
{
        namespace gauss
        {
                enum class kernel_normalization
                {
                        off,
                        on
                };
        }

        ///
        /// \brief constructs a Gaussian filter associated to the given standard deviation.
        /// \param sigma standard deviation of the kernel
        /// \param cutoff threshold to cut/prune the kernel
        /// \param normalize normalize the kernel after cutoff
        ///
        inline vector_t make_gauss_kernel(
                const scalar_t _sigma,
                const scalar_t _cutoff = scalar_t(0.01),
                const gauss::kernel_normalization normalize = gauss::kernel_normalization::on)
        {
                const auto sigma = std::max(scalar_t(1e-6), _sigma);
                const auto cutoff = std::max(scalar_t(1e-6), _cutoff);

                const auto pi = 4 * std::atan(scalar_t(1));
                const auto pi2sqrt = std::sqrt(2 * pi);

                const auto xnorm = scalar_t(0.5) / (sigma * sigma);
                const auto gnorm = scalar_t(1.0) / (pi2sqrt * sigma);

                // estimate radius that produces weights higher than minimum weight <kmin>
                const auto xradius = std::sqrt(-std::log(cutoff / gnorm) / xnorm);
                const int radius = std::max(1, static_cast<int>(xradius));

                // setup kernel
                vector_t kernel(2 * radius + 1);
                for (int x = -radius; x <= radius; x ++)
                {
                        kernel(x + radius) = gnorm * std::exp(-static_cast<scalar_t>(x * x) * xnorm);
                }

                // normalize kernel
                switch (normalize)
                {
                case gauss::kernel_normalization::off:
                        break;

                case gauss::kernel_normalization::on:
                        {
                                const auto wnorm = scalar_t(1) / kernel.sum();
                                kernel *= wnorm;
                        }
                        break;
                }

                return kernel;
        }
}
