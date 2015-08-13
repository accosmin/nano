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
                                                odata.row(r) +=
                                                idata.row(r + kr).segment(0, ocols) * kdata(kr, 0) +
                                                idata.row(r + kr).segment(1, ocols) * kdata(kr, 1) +
                                                idata.row(r + kr).segment(2, ocols) * kdata(kr, 2) +
                                                idata.row(r + kr).segment(3, ocols) * kdata(kr, 3) +
                                                idata.row(r + kr).segment(4, ocols) * kdata(kr, 4) +
                                                idata.row(r + kr).segment(5, ocols) * kdata(kr, 5) +
                                                idata.row(r + kr).segment(6, ocols) * kdata(kr, 6) +
                                                idata.row(r + kr).segment(7, ocols) * kdata(kr, 7) +
                                                idata.row(r + kr).segment(8, ocols) * kdata(kr, 8) +
                                                idata.row(r + kr).segment(9, ocols) * kdata(kr, 9) +
                                                idata.row(r + kr).segment(10, ocols) * kdata(kr, 10) +
                                                idata.row(r + kr).segment(11, ocols) * kdata(kr, 11) +
                                                idata.row(r + kr).segment(12, ocols) * kdata(kr, 12) +
                                                idata.row(r + kr).segment(13, ocols) * kdata(kr, 13) +
                                                idata.row(r + kr).segment(14, ocols) * kdata(kr, 14);
                                        }
                                }
                        }
                };
        }
}

