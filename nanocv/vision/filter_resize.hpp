#pragma once

#include "nanocv/math/cast.hpp"

namespace ncv
{
        ///
        /// \brief resize tensor's planes to the given size (using bilinear interpolation)
        ///
        struct resize_filter_t
        {
                resize_filter_t(const int rows, const int cols)
                        :       m_orows(rows),
                                m_ocols(cols)
                {
                }

                template
                <
                        typename ttensor
                >
                ttensor operator()(const ttensor& src) const
                {
                        const int idims = static_cast<int>(src.dims());
                        const int irows = static_cast<int>(src.rows());
                        const int icols = static_cast<int>(src.cols());
                        const int odims = idims;
                        const int orows = m_orows;
                        const int ocols = m_ocols;

                        const double scale_rows = double(irows) / double(orows);
                        const double scale_cols = double(icols) / double(ocols);

                        const auto op = [=] (const auto& srcplane, auto&& dstplane)
                        {
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

                                                dstplane(_or, _oc) = math::cast<typename ttensor::Scalar>(
                                                        wr0 * wc0 * srcplane(ir0, ic0) +
                                                        wr0 * wc1 * srcplane(ir0, ic1) +
                                                        wr1 * wc1 * srcplane(ir1, ic1) +
                                                        wr1 * wc0 * srcplane(ir1, ic0));
                                        }
                                }
                        };

                        ttensor dst(odims, orows, ocols);

                        for (int _od = 0; _od < odims; _od ++)
                        {
                                op(src.matrix(_od), dst.matrix(_od));
                        }

                        return dst;
                }

                int     m_orows;        ///< output number of rows
                int     m_ocols;        ///< output number of columns
        };
}

