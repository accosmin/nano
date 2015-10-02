#pragma once

#include <cassert>

namespace tensor
{
        ///
        /// \brief 2D convolution: odata += idata @ kdata (using a mad operator for a fixed output size)
        ///
        template
        <
                int ocols
        >
        struct conv2d_madi_t
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
                        assert(odata.cols() == ocols);

                        const auto orows = odata.rows();
                        const auto krows = kdata.rows();
                        const auto kcols = kdata.cols();

                        for (auto r = 0; r < orows; r ++)
                        {
                                auto orow = odata.row(r);

                                for (auto kr = 0; kr < krows; kr ++)
                                {
                                        const auto irow = idata.row(r + kr);

                                        for (auto kc = 0; kc < kcols; kc ++)
                                        {
                                                orow += irow.template segment<ocols>(kc) * kdata(kr, kc);
                                        }
                                }
                        }
                }
        };
}

