#pragma once

#include <cassert>

namespace ncv
{
        namespace tensor
        {
                ///
                /// \brief 2D convolution: odata += idata @ kdata (for 3x3 kernels)
                ///
                struct conv2d_3x3_t
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
                                assert(kdata.rows() == 3);
                                assert(kdata.cols() == 3);

                                const auto orows = odata.rows();
                                const auto ocols = odata.cols();

                                for (auto r = 0; r < orows; r ++)
                                {
                                        odata.row(r) +=

                                        idata.row(r + 0).segment(0, ocols) * kdata(0, 0) +
                                        idata.row(r + 0).segment(1, ocols) * kdata(0, 1) +
                                        idata.row(r + 0).segment(2, ocols) * kdata(0, 2) +

                                        idata.row(r + 1).segment(0, ocols) * kdata(1, 0) +
                                        idata.row(r + 1).segment(1, ocols) * kdata(1, 1) +
                                        idata.row(r + 1).segment(2, ocols) * kdata(1, 2) +

                                        idata.row(r + 2).segment(0, ocols) * kdata(2, 0) +
                                        idata.row(r + 2).segment(1, ocols) * kdata(2, 1) +
                                        idata.row(r + 2).segment(2, ocols) * kdata(2, 2) ;
                                }
                        }
                };
        }
}

