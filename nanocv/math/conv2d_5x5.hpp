#pragma once

#include <cassert>

namespace ncv
{
        namespace math
        {
                ///
                /// \brief 2D convolution: odata += idata @ kdata (for 5x5 kernels)
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
                                        for (int kr = 0; kr < 5; kr ++)
                                        {
                                                odata.row(r) +=
                                                idata.row(r + kr).segment(0, ocols) * kdata(kr, 0) +
                                                idata.row(r + kr).segment(1, ocols) * kdata(kr, 1) +
                                                idata.row(r + kr).segment(2, ocols) * kdata(kr, 2) +
                                                idata.row(r + kr).segment(3, ocols) * kdata(kr, 3) +
                                                idata.row(r + kr).segment(4, ocols) * kdata(kr, 4);
                                        }
                                }
                        }
                };
        }
}

