#pragma once

#include <cassert>

namespace ncv
{
        namespace math
        {
                ///
                /// \brief 2D convolution: odata += idata @ kdata (for 15x15 kernels)
                ///
                struct conv2d_15x15_t
                {
                        template
                        <
                                typename tmatrixi,
                                typename tmatrixk = tmatrixi,
                                typename tmatrixo = tmatrixi,
                                typename tscalar = typename tmatrixo::Scalar
                        >
                        void operator()(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata) const
                        {
                                assert(idata.rows() + 1 == kdata.rows() + odata.rows());
                                assert(idata.cols() + 1 == kdata.cols() + odata.cols());
                                assert(kdata.rows() == 15);
                                assert(kdata.cols() == 15);

                                const auto orows = odata.rows();
                                const auto ocols = odata.cols();

                                for (auto r = 0; r < orows; r ++)
                                {
                                        for (int kr = 0; kr < 15; kr ++)
                                        {
                                                const auto irow = idata.row(r + kr);

                                                odata.row(r) +=
                                                irow.segment(0, ocols) * kdata(kr, 0) +
                                                irow.segment(1, ocols) * kdata(kr, 1) +
                                                irow.segment(2, ocols) * kdata(kr, 2) +
                                                irow.segment(3, ocols) * kdata(kr, 3) +
                                                irow.segment(4, ocols) * kdata(kr, 4);

                                                odata.row(r) +=
                                                irow.segment(5, ocols) * kdata(kr, 5) +
                                                irow.segment(6, ocols) * kdata(kr, 6) +
                                                irow.segment(7, ocols) * kdata(kr, 7) +
                                                irow.segment(8, ocols) * kdata(kr, 8) +
                                                irow.segment(9, ocols) * kdata(kr, 9);

                                                odata.row(r) +=
                                                irow.segment(10, ocols) * kdata(kr, 10) +
                                                irow.segment(11, ocols) * kdata(kr, 11) +
                                                irow.segment(12, ocols) * kdata(kr, 12) +
                                                irow.segment(13, ocols) * kdata(kr, 13) +
                                                irow.segment(14, ocols) * kdata(kr, 14);
                                        }
                                }
                        }
                };
        }
}

