#pragma once

#include <cassert>

namespace tensor
{
        ///
        /// \brief 2D convolution: odata += idata @ kdata (using a dot operator)
        ///
        struct conv2d_dot_t
        {
                template
                <
                        typename tmatrixi,
                        typename tmatrixk,
                        typename tmatrixo
                >
                void operator()(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo&& odata) const
                {
                        assert(idata.rows() + 1 == kdata.rows() + odata.rows());
                        assert(idata.cols() + 1 == kdata.cols() + odata.cols());

                        const auto orows = odata.rows();
                        const auto ocols = odata.cols();
                        const auto krows = kdata.rows();
                        const auto kcols = kdata.cols();

                        for (auto r = 0; r < orows; r ++)
                        {
                                for (auto kr = 0; kr < krows; kr ++)
                                {
                                        const auto irow = idata.row(r + kr);
                                        const auto krow = kdata.row(kr);

                                        for (auto c = 0; c < ocols; c ++)
                                        {
                                                odata(r, c) += irow.segment(c, kcols).dot(krow);
                                        }
                                }
                        }
                }
        };
}

