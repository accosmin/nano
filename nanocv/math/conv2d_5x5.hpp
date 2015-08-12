#pragma once

#include <cassert>

namespace ncv
{
        namespace math
        {
                ///
                /// \brief 2D convolution: odata += idata @ kdata (for 5x5 fixed-size kernels)
                ///
                struct conv2d_5x5_t
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
                                assert(kdata.rows() == 5);
                                assert(kdata.cols() == 5);

                                const auto orows = odata.rows();
                                const auto ocols = odata.cols();

                                for (auto r = 0; r < orows; r ++)
                                {
                                        odata.row(r) +=

                                        idata.row(r + 0).segment(0, ocols) * kdata(0, 0) +
                                        idata.row(r + 0).segment(1, ocols) * kdata(0, 1) +
                                        idata.row(r + 0).segment(2, ocols) * kdata(0, 2) +
                                        idata.row(r + 0).segment(3, ocols) * kdata(0, 3) +
                                        idata.row(r + 0).segment(4, ocols) * kdata(0, 4) +

                                        idata.row(r + 1).segment(0, ocols) * kdata(1, 0) +
                                        idata.row(r + 1).segment(1, ocols) * kdata(1, 1) +
                                        idata.row(r + 1).segment(2, ocols) * kdata(1, 2) +
                                        idata.row(r + 1).segment(3, ocols) * kdata(1, 3) +
                                        idata.row(r + 1).segment(4, ocols) * kdata(1, 4) +

                                        idata.row(r + 2).segment(0, ocols) * kdata(2, 0) +
                                        idata.row(r + 2).segment(1, ocols) * kdata(2, 1) +
                                        idata.row(r + 2).segment(2, ocols) * kdata(2, 2) +
                                        idata.row(r + 2).segment(3, ocols) * kdata(2, 3) +
                                        idata.row(r + 2).segment(4, ocols) * kdata(2, 4) +

                                        idata.row(r + 3).segment(0, ocols) * kdata(3, 0) +
                                        idata.row(r + 3).segment(1, ocols) * kdata(3, 1) +
                                        idata.row(r + 3).segment(2, ocols) * kdata(3, 2) +
                                        idata.row(r + 3).segment(3, ocols) * kdata(3, 3) +
                                        idata.row(r + 3).segment(4, ocols) * kdata(3, 4) +

                                        idata.row(r + 4).segment(0, ocols) * kdata(4, 0) +
                                        idata.row(r + 4).segment(1, ocols) * kdata(4, 1) +
                                        idata.row(r + 4).segment(2, ocols) * kdata(4, 2) +
                                        idata.row(r + 4).segment(3, ocols) * kdata(4, 3) +
                                        idata.row(r + 4).segment(4, ocols) * kdata(4, 4);
                                }
                        }
                };
        }
}

