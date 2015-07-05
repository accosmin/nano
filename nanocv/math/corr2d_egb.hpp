#pragma once

#include <cassert>

namespace ncv
{
        namespace math
        {
                ///
                /// \brief 2D correlation: idata += odata @ kdata (using Eigen 2D blocks)
                ///
                struct corr2d_egb_t
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
                                        for (auto c = 0; c < odata.cols(); c ++)
                                        {
                                                idata.block(r, c, kdata.rows(), kdata.cols()) +=
                                                kdata * odata(r, c);
                                        }
                                }
                        }
                };
        }
}

