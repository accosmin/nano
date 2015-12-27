#pragma once

#include <cassert>

namespace tensor
{
        ///
        /// \brief 2D correlation: idata += odata @ kdata (using a fixed-size mad product by odata columns)
        ///
        template
        <
                int ocols
        >
        struct corr2d_mdoi_t
        {
                template
                <
                        typename tmatrixo,
                        typename tmatrixk,
                        typename tmatrixi,
                        typename tscalar = typename tmatrixi::Scalar
                >
                void operator()(const tmatrixo& odata, const tmatrixk& kdata, tmatrixi&& idata) const
                {
                        assert(idata.rows() + 1 == kdata.rows() + odata.rows());
                        assert(idata.cols() + 1 == kdata.cols() + odata.cols());
                        assert(odata.cols() == ocols);

                        const auto orows = odata.rows();
                        const auto krows = kdata.rows();
                        const auto kcols = kdata.cols();

                        for (auto r = 0; r < orows; r ++)
                        {
                                for (auto kr = 0; kr < krows; kr ++)
                                {
                                        const auto orow = odata.row(r);

                                        for (auto kc = 0; kc < kcols; kc ++)
                                        {
                                                idata.row(r + kr).template segment<ocols>(kc) +=
                                                orow * kdata(kr, kc);
                                        }
                                }
                        }
                }
        };
}

