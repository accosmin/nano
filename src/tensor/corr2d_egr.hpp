#pragma once

#include <cassert>

namespace tensor
{
        ///
        /// \brief 2D correlation: idata += odata @ kdata (using Eigen 1D row-blocks)
        ///
        struct corr2d_egr_t
        {
                template
                <
                        typename tmatrixo,
                        typename tmatrixk = tmatrixo,
                        typename tmatrixi = tmatrixo
                >
                void operator()(const tmatrixo& odata, const tmatrixk& kdata, tmatrixi& idata) const
                {
                        assert(idata.rows() + 1 == kdata.rows() + odata.rows());
                        assert(idata.cols() + 1 == kdata.cols() + odata.cols());

                        for (auto r = 0; r < odata.rows(); r ++)
                        {
                                for (auto kr = 0; kr < kdata.rows(); kr ++)
                                {
                                        for (auto kc = 0; kc < kdata.cols(); kc ++)
                                        {
                                                idata.block(r + kr, kc, 1, odata.cols()) +=
                                                odata.row(r) * kdata(kr, kc);
                                        }
                                }
                        }
                }
        };
}
