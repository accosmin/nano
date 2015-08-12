#pragma once

#include <cassert>

namespace ncv
{
        namespace math
        {
                ///
                /// \brief 2D convolution: odata += idata @ kdata (using plain array indexing)
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

                                const auto k00 = kdata(0, 0), k01 = kdata(0, 1), k02 = kdata(0, 2);
                                const auto k10 = kdata(1, 0), k11 = kdata(1, 1), k12 = kdata(1, 2);
                                const auto k20 = kdata(2, 0), k21 = kdata(2, 1), k22 = kdata(2, 2);

                                for (auto r = 0; r < orows; r ++)
                                {
                                        for (auto c = 0; c < ocols; c ++)
                                        {
                                                const tscalar sum00 = idata(r + 0, c + 0) * k00;
                                                const tscalar sum01 = idata(r + 0, c + 1) * k01;
                                                const tscalar sum02 = idata(r + 0, c + 2) * k02;

                                                const tscalar sum10 = idata(r + 1, c + 0) * k10;
                                                const tscalar sum11 = idata(r + 1, c + 1) * k11;
                                                const tscalar sum12 = idata(r + 1, c + 2) * k12;

                                                const tscalar sum20 = idata(r + 2, c + 0) * k20;
                                                const tscalar sum21 = idata(r + 2, c + 1) * k21;
                                                const tscalar sum22 = idata(r + 2, c + 2) * k22;

                                                odata(r, c) +=
                                                sum00 + sum01 + sum02 +
                                                sum10 + sum11 + sum12 +
                                                sum20 + sum21 + sum22;
                                        }
                                }
                        }
                };
        }
}

