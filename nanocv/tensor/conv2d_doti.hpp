#pragma once

#include <cassert>

namespace tensor
{
        ///
        /// \brief 2D convolution: odata += idata @ kdata (using a dot operator with a fixed kernel size)
        ///
        template
        <
                int kcols
        >
        struct conv2d_doti_t
        {
                template
                <
                        typename tmatrixi,
                        typename tmatrixk = tmatrixi,
                        typename tmatrixo = tmatrixi
                >
                void operator()(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata) const
                {
                        assert(idata.rows() + 1 == kdata.rows() + odata.rows());
                        assert(idata.cols() + 1 == kdata.cols() + odata.cols());
                        assert(kdata.cols() == kcols);

                        const auto orows = odata.rows();
                        const auto ocols = odata.cols();
                        const auto krows = kdata.rows();

                        for (auto r = 0; r < orows; r ++)
                        {
                                for (auto kr = 0; kr < krows; kr ++)
                                {
                                        const auto irow = idata.row(r + kr);
                                        const auto krow = kdata.row(kr).template segment<kcols>(0);

                                        for (auto c = 0; c < ocols; c ++)
                                        {
                                                odata(r, c) += irow.template segment<kcols>(c).dot(krow);
                                        }
                                }
                        }
                }
        };
}

