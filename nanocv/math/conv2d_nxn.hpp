#pragma once

#include <cassert>

namespace ncv
{
        namespace math
        {
                ///
                /// \brief 2D convolution: odata += idata @ kdata (for NxN kernels)
                ///
                template
                <
                        int tksize
                >
                struct conv2d_nxn_t
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
                                assert(kdata.rows() == tksize);
                                assert(kdata.cols() == tksize);

                                const auto orows = odata.rows();
                                const auto ocols = odata.cols();

                                for (auto r = 0; r < orows; r ++)
                                {
                                        for (int kr = 0; kr < tksize; kr ++)
                                        {
                                                int kc = 0;
                                                for ( ; kc + 2 < tksize; kc += 3)
                                                {
                                                        odata.row(r) +=
                                                        idata.row(r + kr).segment(kc + 0, ocols) * kdata(kr, kc + 0) +
                                                        idata.row(r + kr).segment(kc + 1, ocols) * kdata(kr, kc + 1) +
                                                        idata.row(r + kr).segment(kc + 2, ocols) * kdata(kr, kc + 2);
                                                }
                                                for ( ; kc < tksize; kc ++)
                                                {
                                                        odata.row(r) +=
                                                        idata.row(r + kr).segment(kc + 0, ocols) * kdata(kr, kc + 0);
                                                }
                                        }
                                }
                        }
                };
        }
}

