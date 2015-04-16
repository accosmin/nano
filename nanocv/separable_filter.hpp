#pragma once

#include "math/range.hpp"
#include <vector>

namespace ncv
{
        ///
        /// \brief in-place separable 2D filter
        ///
        template
        <
                typename tkernel,                       ///< 1D kernel that composes the 2D filter
                typename tmatrix,                       ///< source data
                typename tgetter,                       ///< extract value from element (e.g. pixel)
                typename tsetter,                       ///< set value to element (e.g. pixel)

                typename tscalar = typename tkernel::tscalar,
                typename tvalue = typename tmatrix::Scalar
        >
        bool inplace_separable_filter(const tkernel& kernel, const range_t<tscalar>& range,
                tmatrix& src, tgetter getter, tsetter setter)
        {
                const int rows = static_cast<int>(src.rows());
                const int cols = static_cast<int>(src.cols());

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
                                buff[c] = math::cast<tscalar>(getter(src(r, c)));
                        }

                        for (int c = 0; c < cols; c ++)
                        {
                                tscalar v = 0;
                                for (int k = -krad; k <= krad; k ++)
                                {
                                        const int cc = math::clamp(k + c, 0, cols - 1);
                                        v += kernel[k + krad] * buff[cc];
                                }

                                src(r, c) = setter(src(r, c), math::cast<tvalue>(range.clamp(v)));
                        }
                }

                // vertical filter
                for (int c = 0; c < cols; c ++)
                {
                        for (int r = 0; r < rows; r ++)
                        {
                                buff[r] = math::cast<tscalar>(getter(src(r, c)));
                        }

                        for (int r = 0; r < rows; r ++)
                        {
                                tscalar v = 0;
                                for (int k = -krad; k <= krad; k ++)
                                {
                                        const int rr = math::clamp(k + r, 0, rows - 1);
                                        v += kernel[k + krad] * buff[rr];
                                }

                                src(r, c) = setter(src(r, c), math::cast<tvalue>(range.clamp(v)));
                        }
                }

                // OK
                return true;
        }
}

