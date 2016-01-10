#pragma once

#include <cassert>

namespace tensor
{
        ///
        /// \brief 2D correlation: idata += odata @ kdata (using a mad product by kdata columns)
        ///
        struct corr2d_mdk_t
        {
                template
                <
                        typename tmatrixo,
                        typename tmatrixk,
                        typename tmatrixi
                >
                void operator()(const tmatrixo& odata, const tmatrixk& kdata, tmatrixi&& idata) const
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
                                        const auto krow = kdata.row(kr);

                                        for (auto c = 0; c < ocols; c ++)
                                        {
                                                idata.row(r + kr).segment(c, kcols) +=
                                                krow * odata(r, c);
                                        }
                                }
                        }
                }
        };

        ///
        /// \brief 2D correlation: idata += odata @ kdata (using a fixed-size mad product by kdata columns)
        ///
        template
        <
                int kcols
        >
        struct corr2d_mdki_t
        {
                template
                <
                        typename tmatrixo,
                        typename tmatrixk,
                        typename tmatrixi
                >
                void operator()(const tmatrixo& odata, const tmatrixk& kdata, tmatrixi&& idata) const
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
                                        const auto krow = kdata.row(kr);

                                        for (auto c = 0; c < ocols; c ++)
                                        {
                                                idata.row(r + kr).template segment<kcols>(c) +=
                                                krow * odata(r, c);
                                        }
                                }
                        }
                }
        };

        ///
        /// \brief 2D correlation: idata += odata @ kdata (using a mad product by kdata runtime-decoded columns)
        ///
        struct corr2d_mdk_dyn_t
        {
                template
                <
                        typename tmatrixo,
                        typename tmatrixk,
                        typename tmatrixi
                >
                void operator()(const tmatrixo& odata, const tmatrixk& kdata, tmatrixi&& idata) const
                {
                        switch (kdata.cols())
                        {
                        case 1:         corr2d_mdki_t<1>()(odata, kdata, idata); break;
                        case 2:         corr2d_mdki_t<2>()(odata, kdata, idata); break;
                        case 3:         corr2d_mdki_t<3>()(odata, kdata, idata); break;
                        case 4:         corr2d_mdki_t<4>()(odata, kdata, idata); break;
                        case 5:         corr2d_mdki_t<5>()(odata, kdata, idata); break;
                        case 6:         corr2d_mdki_t<6>()(odata, kdata, idata); break;
                        case 7:         corr2d_mdki_t<7>()(odata, kdata, idata); break;
                        case 8:         corr2d_mdki_t<8>()(odata, kdata, idata); break;
                        case 9:         corr2d_mdki_t<9>()(odata, kdata, idata); break;
                        case 10:        corr2d_mdki_t<10>()(odata, kdata, idata); break;
                        case 11:        corr2d_mdki_t<11>()(odata, kdata, idata); break;
                        case 12:        corr2d_mdki_t<12>()(odata, kdata, idata); break;
                        case 13:        corr2d_mdki_t<13>()(odata, kdata, idata); break;
                        case 14:        corr2d_mdki_t<14>()(odata, kdata, idata); break;
                        case 15:        corr2d_mdki_t<15>()(odata, kdata, idata); break;
                        default:        corr2d_mdk_t()(odata, kdata, idata); break;
                        }
                }
        };
}

