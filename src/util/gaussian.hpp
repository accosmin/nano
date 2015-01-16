#pragma once

#include "gauss.hpp"

namespace ncv
{
        ///
        /// \brief filter the input matrix with a Gaussian kernel having the given standard deviation sigma
        ///
        template
        <
                typename tscalar = double,
                typename tmatrix,
                typename tgetter,                       ///< extract value from element (e.g. pixel)
                typename tsetter,                       ///< set value to element (e.g. pixel)

                typename tvalue = typename tmatrix::Scalar
        >
        bool gaussian(const tmatrix& src, tmatrix& dst, tscalar sigma, tvalue minv, tvalue maxv, tgetter getter, tsetter setter)
        {
                const int rows = static_cast<int>(src.rows());
                const int cols = static_cast<int>(src.cols());

                if (&src != &dst)
                {
                        dst = src;
                }

                // construct Gaussian kernel
                const std::vector<tscalar> kernel = make_gaussian(sigma);

                const int ksize = static_cast<int>(kernel.size());
                const int krad = ksize / 2;

                if (ksize != (2 * krad + 1))
                {
                        return false;
                }

                std::vector<tscalar> buff(std::max(rows, cols));

                // horizontal filter
                for (int r = 0; r < rows; r ++)
                {
                        for (int c = 0; c < cols; c ++)
                        {
                                buff[c] = math::cast<tscalar>(getter(dst(r, c)));
                        }

                        for (int c = 0; c < cols; c ++)
                        {
                                tscalar v = 0;
                                for (int k = -krad; k <= krad; k ++)
                                {
                                        const int cc = math::clamp(k + c, 0, cols - 1);
                                        v += kernel[k + krad] * buff[cc];
                                }

                                dst(r, c) = setter(dst(r, c), math::cast<tvalue>(math::clamp(v, minv, maxv)));
                        }
                }

                // vertical filter
                for (int c = 0; c < cols; c ++)
                {
                        for (int r = 0; r < rows; r ++)
                        {
                                buff[r] = math::cast<tscalar>(getter(dst(r, c)));
                        }

                        for (int r = 0; r < rows; r ++)
                        {
                                tscalar v = 0;
                                for (int k = -krad; k <= krad; k ++)
                                {
                                        const int rr = math::clamp(k + r, 0, rows - 1);
                                        v += kernel[k + krad] * buff[rr];
                                }

                                dst(r, c) = setter(dst(r, c), math::cast<tvalue>(math::clamp(v, minv, maxv)));
                        }
                }

                return true;
        }
}

