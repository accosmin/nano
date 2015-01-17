#pragma once

#include "math.hpp"
#include "cast.hpp"

namespace ncv
{
        ///
        /// \brief resize the input matrix by the given factor (using bilinear interpolation)
        ///
        template
        <
                typename tscalar = double,
                typename tmatrix,
                typename tmixer,                        ///< interpolate (pixel) values given weights

                typename tvalue = typename tmatrix::Scalar
        >
        bool bilinear(const tmatrix& src, tmatrix& dst, tscalar factor, tmixer mixer)
        {
                static const tscalar eps = math::cast<tscalar>(1e-2);
                static const tscalar one = math::cast<tscalar>(1.0);

                if (factor < eps || math::abs(factor - one) < eps)
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

                                        dst(_or, _oc) = mixer(
                                                wr0 * wc0, src(ir0, ic0),
                                                wr0 * wc1, src(ir0, ic1),
                                                wr1 * wc1, src(ir1, ic1),
                                                wr1 * wc0, src(ir1, ic0));
                                }
                        }
                }

                // OK
                return true;
        }
}

