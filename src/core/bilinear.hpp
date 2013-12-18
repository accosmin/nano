#ifndef NANOCV_BILINEAR_H
#define NANOCV_BILINEAR_H

#include <algorithm>
#include "math.hpp"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // bilinear interpolation.
        /////////////////////////////////////////////////////////////////////////////////////////

        namespace math
        {
                // resize the input matrix by the given factor (using bilinear interpolation)
                template
                <       typename tmatrix,
                        typename tscalar = double
                >
                void bilinear(const tmatrix& src, tmatrix& dst, tscalar factor)
                {
                        typedef typename tmatrix::Scalar tvalue;

                        static const tscalar eps = math::cast<tscalar>(1e-2);
                        static const tscalar one = math::cast<tscalar>(1.0);

                        if (factor < eps || std::abs(factor - one) < eps)
                        {
                                dst = src;
                        }
                        else
                        {
                                const int irows = math::cast<int>(src.rows());
                                const int icols = math::cast<int>(src.cols());
                                const int orows = math::cast<int>(0.5 + factor * irows);
                                const int ocols = math::cast<int>(0.5 + factor * icols);
                                const tscalar is = one / factor;

                                // bilinear interpolation
                                dst.resize(orows, ocols);
                                for (int _or = 0; _or < orows; _or ++)
                                {
                                        const tscalar isr = is * _or;
                                        const int ir0 = math::cast<int>(isr), ir1 = std::min(ir0 + 1, irows - 1);
                                        const tscalar wr1 = isr - ir0, wr0 = one - wr1;

                                        for (int _oc = 0; _oc < ocols; _oc ++)
                                        {
                                                const tscalar isc = is * _oc;
                                                const int ic0 = math::cast<int>(isc), ic1 = std::min(ic0 + 1, icols - 1);
                                                const tscalar wc1 = isc - ic0, wc0 = one - wc1;

                                                dst(_or, _oc) = cast<tvalue>(
                                                        wr0 * wc0 * src(ir0, ic0) +
                                                        wr0 * wc1 * src(ir0, ic1) +
                                                        wr1 * wc1 * src(ir1, ic1) +
                                                        wr1 * wc0 * src(ir1, ic0));
                                        }
                                }
                        }
                }

                // resize the input matrix to the given maximum matrix size
                template
                <
                        typename tmatrix,
                        typename tsize = int
                >
                void bilinear(const tmatrix& src, tmatrix& dst, tsize max_rows, tsize max_cols)
                {
                        const tsize rows = math::cast<tsize>(src.rows());
                        const tsize cols = math::cast<tsize>(src.cols());
                        const tsize one = math::cast<tsize>(1);

                        if (    rows < one || max_rows < one ||
                                cols < one || max_cols < one)
                        {
                                dst = src;
                        }
                        else
                        {
                                const double srows = cast<double>(max_rows) / cast<double>(rows);
                                const double scols = cast<double>(max_cols) / cast<double>(cols);
                                bilinear(src, dst, std::min(srows, scols));
                        }
                }
        }
}

#endif // NANOCV_BILINEAR_H

