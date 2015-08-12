#pragma once

#include <cassert>

namespace ncv
{
        namespace math
        {
                ///
                /// \brief 2D convolution: odata += idata @ kdata (using plain array indexing)
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
                                        for (auto c = 0; c < ocols; c ++)
                                        {
                                                const tscalar sum00 = idata(r + 0, c + 0) * kdata(0, 0);
                                                const tscalar sum01 = idata(r + 0, c + 1) * kdata(0, 1);
                                                const tscalar sum02 = idata(r + 0, c + 2) * kdata(0, 2);
                                                const tscalar sum03 = idata(r + 0, c + 3) * kdata(0, 3);
                                                const tscalar sum04 = idata(r + 0, c + 4) * kdata(0, 4);

                                                const tscalar sum10 = idata(r + 1, c + 0) * kdata(1, 0);
                                                const tscalar sum11 = idata(r + 1, c + 1) * kdata(1, 1);
                                                const tscalar sum12 = idata(r + 1, c + 2) * kdata(1, 2);
                                                const tscalar sum13 = idata(r + 1, c + 3) * kdata(1, 3);
                                                const tscalar sum14 = idata(r + 1, c + 4) * kdata(1, 4);

                                                const tscalar sum20 = idata(r + 2, c + 0) * kdata(2, 0);
                                                const tscalar sum21 = idata(r + 2, c + 1) * kdata(2, 1);
                                                const tscalar sum22 = idata(r + 2, c + 2) * kdata(2, 2);
                                                const tscalar sum23 = idata(r + 2, c + 3) * kdata(2, 3);
                                                const tscalar sum24 = idata(r + 2, c + 4) * kdata(2, 4);

                                                const tscalar sum30 = idata(r + 3, c + 0) * kdata(3, 0);
                                                const tscalar sum31 = idata(r + 3, c + 1) * kdata(3, 1);
                                                const tscalar sum32 = idata(r + 3, c + 2) * kdata(3, 2);
                                                const tscalar sum33 = idata(r + 3, c + 3) * kdata(3, 3);
                                                const tscalar sum34 = idata(r + 3, c + 4) * kdata(3, 4);

                                                const tscalar sum40 = idata(r + 4, c + 0) * kdata(4, 0);
                                                const tscalar sum41 = idata(r + 4, c + 1) * kdata(4, 1);
                                                const tscalar sum42 = idata(r + 4, c + 2) * kdata(4, 2);
                                                const tscalar sum43 = idata(r + 4, c + 3) * kdata(4, 3);
                                                const tscalar sum44 = idata(r + 4, c + 4) * kdata(4, 4);

                                                odata(r, c) +=
                                                sum00 + sum01 + sum02 + sum03 + sum04 +
                                                sum10 + sum11 + sum12 + sum13 + sum14 +
                                                sum20 + sum21 + sum22 + sum23 + sum24 +
                                                sum30 + sum31 + sum32 + sum33 + sum34 +
                                                sum40 + sum41 + sum42 + sum43 + sum44;
                                        }
                                }
                        }
                };
        }
}

