#pragma once

#include <cassert>
#include <algorithm>
#include "math/cast.hpp"

namespace cortex
{
        ///
        /// \brief resize the input matrix to the output matrix (using bilinear interpolation)
        ///
        /// NB: the output matrix is considered to be already resized to the desired size.
        ///
        template
        <
                typename tmatrixi,
                typename tmatrixo
        >
        void bilinear(const tmatrixi& srcplane, tmatrixo&& dstplane)
        {
                assert(srcplane.rows() > 0);
                assert(srcplane.cols() > 0);

                const int irows = static_cast<int>(srcplane.rows());
                const int icols = static_cast<int>(srcplane.cols());
                const int orows = static_cast<int>(dstplane.rows());
                const int ocols = static_cast<int>(dstplane.cols());

                const double scale_rows = double(irows) / double(orows);
                const double scale_cols = double(icols) / double(ocols);

                for (int _or = 0; _or < orows; _or ++)
                {
                        const double isr = scale_rows * _or;
                        const int ir0 = static_cast<int>(isr), ir1 = std::min(ir0 + 1, irows - 1);
                        const double wr1 = isr - ir0, wr0 = 1.0 - wr1;

                        for (int _oc = 0; _oc < ocols; _oc ++)
                        {
                                const double isc = scale_cols * _oc;
                                const int ic0 = static_cast<int>(isc), ic1 = std::min(ic0 + 1, icols - 1);
                                const double wc1 = isc - ic0, wc0 = 1.0 - wc1;

                                dstplane(_or, _oc) = math::cast<typename tmatrixo::Scalar>(
                                        wr0 * wc0 * srcplane(ir0, ic0) +
                                        wr0 * wc1 * srcplane(ir0, ic1) +
                                        wr1 * wc1 * srcplane(ir1, ic1) +
                                        wr1 * wc0 * srcplane(ir1, ic0));
                        }
                }
        }
}

