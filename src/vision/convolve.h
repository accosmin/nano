#pragma once

#include <cassert>
#include "tensor/vector.h"

namespace nano
{
        ///
        /// \brief in-place separable 2D filter.
        ///
        template <typename tmatrix>
        void convolve(const vector_t& kernel, tmatrix&& src)
        {
                const int rows = static_cast<int>(src.rows());
                const int cols = static_cast<int>(src.cols());
                assert(rows > 1 && cols > 1);

                const int ksize = static_cast<int>(kernel.size());
                const int krad = ksize / 2;
                assert(ksize == 2 * krad + 1);

                vector_t buff(std::max(rows, cols) + ksize);

                // horizontal filter
                for (int r = 0; r < rows; ++ r)
                {
                        buff.segment(0, krad).setConstant(src(r, 0));
                        buff.segment(krad, cols) = src.row(r);
                        buff.segment(krad + cols, krad).setConstant(src(r, cols - 1));

                        for (int c = 0; c < cols; ++ c)
                        {
                                src(r, c) = buff.segment(c - krad, ksize).dot(kernel);
                        }
                }

                // vertical filter
                for (int c = 0; c < cols; ++ c)
                {
                        buff.segment(0, krad).setConstant(src(0, c));
                        buff.segment(krad, rows) = src.col(c);
                        buff.segment(krad + rows, krad).setConstant(src(rows - 1, c));

                        for (int r = 0; r < rows; ++ r)
                        {
                                src(r, c) = buff.segment(r - krad, ksize).dot(kernel);
                        }
                }
        }
}
