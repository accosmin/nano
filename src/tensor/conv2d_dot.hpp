#pragma once

#include "conv2d_assert.hpp"

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
                        conv2d_assert(idata, kdata, odata);

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
                        typename tmatrixk,
                        typename tmatrixo
                >
                void operator()(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo&& odata) const
                {
                        conv2d_assert(idata, kdata, odata);
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

        ///
        /// \brief 2D convolution: odata += idata @ kdata (using a dot operator with a runtime-decoded kernel size)
        ///
        struct conv2d_dot_dyn_t
        {
                template
                <
                        typename tmatrixi,
                        typename tmatrixk,
                        typename tmatrixo
                >
                void operator()(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo&& odata) const
                {
                        switch (kdata.cols())
                        {
                        case 1:         conv2d_doti_t<1>()(idata, kdata, odata); break;
                        case 2:         conv2d_doti_t<2>()(idata, kdata, odata); break;
                        case 3:         conv2d_doti_t<3>()(idata, kdata, odata); break;
                        case 4:         conv2d_doti_t<4>()(idata, kdata, odata); break;
                        case 5:         conv2d_doti_t<5>()(idata, kdata, odata); break;
                        case 6:         conv2d_doti_t<6>()(idata, kdata, odata); break;
                        case 7:         conv2d_doti_t<7>()(idata, kdata, odata); break;
                        case 8:         conv2d_doti_t<8>()(idata, kdata, odata); break;
                        case 9:         conv2d_doti_t<9>()(idata, kdata, odata); break;
                        case 10:        conv2d_doti_t<10>()(idata, kdata, odata); break;
                        case 11:        conv2d_doti_t<11>()(idata, kdata, odata); break;
                        case 12:        conv2d_doti_t<12>()(idata, kdata, odata); break;
                        case 13:        conv2d_doti_t<13>()(idata, kdata, odata); break;
                        case 14:        conv2d_doti_t<14>()(idata, kdata, odata); break;
                        case 15:        conv2d_doti_t<15>()(idata, kdata, odata); break;
                        case 16:        conv2d_doti_t<16>()(idata, kdata, odata); break;
                        case 17:        conv2d_doti_t<17>()(idata, kdata, odata); break;
                        case 18:        conv2d_doti_t<18>()(idata, kdata, odata); break;
                        case 19:        conv2d_doti_t<19>()(idata, kdata, odata); break;
                        case 20:        conv2d_doti_t<20>()(idata, kdata, odata); break;
                        case 21:        conv2d_doti_t<21>()(idata, kdata, odata); break;
                        case 22:        conv2d_doti_t<22>()(idata, kdata, odata); break;
                        case 23:        conv2d_doti_t<23>()(idata, kdata, odata); break;
                        case 24:        conv2d_doti_t<24>()(idata, kdata, odata); break;
                        case 25:        conv2d_doti_t<25>()(idata, kdata, odata); break;
                        case 26:        conv2d_doti_t<26>()(idata, kdata, odata); break;
                        case 27:        conv2d_doti_t<27>()(idata, kdata, odata); break;
                        case 28:        conv2d_doti_t<28>()(idata, kdata, odata); break;
                        case 29:        conv2d_doti_t<29>()(idata, kdata, odata); break;
                        case 30:        conv2d_doti_t<30>()(idata, kdata, odata); break;
                        default:        conv2d_dot_t()(idata, kdata, odata); break;
                        }
                }
        };
}

