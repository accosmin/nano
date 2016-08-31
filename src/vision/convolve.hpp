#pragma once

#include <cassert>
#include <algorithm>
#include "math/clamp.hpp"
#include "tensor/vector.hpp"

namespace nano
{
        ///
        /// \brief in-place separable 2D filter
        ///
        template
        <
                typename tkernel,                       ///< 1D kernel that composes the 2D filter
                typename tmatrix,                       ///< source data

                typename tscalar = typename tkernel::tscalar,
                typename tvalue = typename std::remove_reference<tmatrix>::type::Scalar
        >
        void convolve(const tkernel& kernel, tmatrix&& src)
        {
                const int rows = static_cast<int>(src.rows());
                const int cols = static_cast<int>(src.cols());

                const int ksize = static_cast<int>(kernel.size());
                const int krad = ksize / 2;

                assert(ksize == 2 * krad + 1);

                tensor::vector_t<tscalar> buff(std::max(rows, cols));

                // horizontal filter
                for (int r = 0; r < rows; ++ r)
                {
                        for (int c = 0; c < cols; ++ c)
                        {
                                buff(c) = static_cast<tscalar>(src(r, c));
                        }

                        for (int c = 0; c < cols; ++ c)
                        {
                                tscalar v = 0;
                                for (int k = -krad; k <= krad; ++ k)
                                {
                                        const int cc = nano::clamp(k + c, 0, cols - 1);
                                        v += kernel[k + krad] * buff(cc);
                                }

                                src(r, c) = static_cast<tvalue>(v);
                        }
                }

                // vertical filter
                for (int c = 0; c < cols; ++ c)
                {
                        for (int r = 0; r < rows; ++ r)
                        {
                                buff(r) = static_cast<tscalar>(src(r, c));
                        }

                        for (int r = 0; r < rows; ++ r)
                        {
                                tscalar v = 0;
                                for (int k = -krad; k <= krad; ++ k)
                                {
                                        const int rr = nano::clamp(k + r, 0, rows - 1);
                                        v += kernel[k + krad] * buff(rr);
                                }

                                src(r, c) = static_cast<tvalue>(v);
                        }
                }
        }
}

