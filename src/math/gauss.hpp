#pragma once

#include "cast.hpp"
#include <vector>
#include <numeric>
#include <algorithm>
#include <type_traits>

namespace math
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
        /// \brief constructs a Gaussian filter associated to the given standard deviation
        ///
        template
        <
                typename tscalar_,

                /// disable for not valid types!
                typename tvalid_tscalar = typename std::enable_if<std::is_floating_point<tscalar_>::value>::type
        >
        class gauss_kernel_t
        {
        public:

                using tscalar = tscalar_;

                ///
                /// \brief constructor
                /// \param sigma standard deviation of the kernel
                /// \param cutoff threshold to cut/prune the kernel
                /// \param normalize normalize the kernel after cutoff
                ///
                gauss_kernel_t(tscalar sigma,
                               tscalar cutoff = tscalar(0.01),
                               gauss::kernel_normalization normalize = gauss::kernel_normalization::on)
                {
                        setup(sigma, cutoff, normalize);
                }

                ///
                /// \brief kernel size (always of odd size)
                ///
                std::size_t size() const { return m_kernel.size(); }

                ///
                /// \brief kernel
                ///
                const std::vector<tscalar>& kernel() const { return m_kernel; }

                ///
                /// \brief operator []
                ///
                tscalar operator[](std::size_t index) const { return m_kernel[index]; }

                ///
                /// \brief sum the kernel's elements
                ///
                tscalar sum() const { return std::accumulate(m_kernel.begin(), m_kernel.end(), tscalar(0)); }

        private:

                void setup(tscalar _sigma, tscalar _cutoff, gauss::kernel_normalization normalize)
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
                        m_kernel.resize(static_cast<std::size_t>(2 * radius + 1));
                        for (int x = -radius; x <= radius; x ++)
                        {
                                m_kernel[static_cast<size_t>(x + radius)] =
                                math::cast<tscalar>(gnorm * std::exp(-x * x * xnorm));
                        }

                        // normalize kernel
                        switch (normalize)
                        {
                        case gauss::kernel_normalization::off:
                                break;

                        case gauss::kernel_normalization::on:
                                {
                                        const tscalar wnorm = tscalar(1) / sum();
                                        std::for_each(m_kernel.begin(), m_kernel.end(), [=] (tscalar& w)
                                        {
                                                w *= wnorm;
                                        });
                                }
                                break;
                        }
                }

        private:

                std::vector<tscalar>    m_kernel;       ///< (normalized) 1D Gaussian kernel
        };
}

