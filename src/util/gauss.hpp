#pragma once

#include "cast.hpp"
#include <vector>
#include <type_traits>
#include <numeric>

namespace ncv
{
        ///
        /// \brief compute the Gaussian filter associated to the given standard deviation
        ///
        /// \param sigma variance of the kernel
        /// \param cutoff threshold to cut/prune the kernel
        ///
        template
        <
                typename tscalar,

                /// disable for not valid types!
                typename tvalid_tscalar = typename std::enable_if<std::is_floating_point<tscalar>::value>::type
        >
        std::vector<tscalar> make_gaussian(tscalar _sigma, tscalar _cutoff, bool normalize = true)
        {
                const double sigma = std::max(1e-6, static_cast<double>(_sigma));
                const double cutoff = std::max(1e-6, static_cast<double>(_cutoff));

                static const double pi = 4.0 * std::atan2(1.0, 1.0);
                static const double pi2sqrt = std::sqrt(2.0 * pi);

                const double xnorm = 0.5 / (sigma * sigma);
                const double gnorm = 1.0 / (pi2sqrt * sigma);

                // estimate radius that produces weights higher than minimum weight <kmin>
                const double xradius = std::sqrt(-std::log(cutoff / gnorm) / xnorm);
                const int radius = std::max(1, math::cast<int>(xradius));

                // setup kernel
                std::vector<tscalar> kernel(2 * radius + 1);
                for (int x = -radius; x <= radius; x ++)
                {
                        kernel[x + radius] = math::cast<tscalar>(gnorm * std::exp(-x * x * xnorm));
                }

                // normalize kernel
                if (normalize)
                {
                        const tscalar wnorm = tscalar(1) / std::accumulate(kernel.begin(), kernel.end(), tscalar(0));
                        std::for_each(kernel.begin(), kernel.end(), [=] (tscalar& w) { w *= wnorm; });
                }

                // OK
                return kernel;
        }
}

