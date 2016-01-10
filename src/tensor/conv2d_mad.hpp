#pragma once

#include <cassert>

namespace tensor
{
        ///
        /// \brief 2D convolution: odata += idata @ kdata (using a mad operator)
        ///
        struct conv2d_mad_t
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
                                auto orow = odata.row(r);

                                for (auto kr = 0; kr < krows; kr ++)
                                {
                                        const auto irow = idata.row(r + kr);

                                        for (auto kc = 0; kc < kcols; kc ++)
                                        {
                                                orow += irow.segment(kc, ocols) * kdata(kr, kc);
                                        }
                                }
                        }
                }
        };

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
                        typename tmatrixk,
                        typename tmatrixo
                >
                void operator()(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo&& odata) const
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

        ///
        /// \brief 2D convolution: odata += idata @ kdata (using a mad operator for a runtime-decoded output size)
        ///
        struct conv2d_mad_dyn_t
        {
                template
                <
                        typename tmatrixi,
                        typename tmatrixk,
                        typename tmatrixo
                >
                void operator()(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo&& odata) const
                {
                        switch (odata.cols())
                        {
                        case 1:         conv2d_madi_t<1>()(idata, kdata, odata); break;
                        case 2:         conv2d_madi_t<2>()(idata, kdata, odata); break;
                        case 3:         conv2d_madi_t<3>()(idata, kdata, odata); break;
                        case 4:         conv2d_madi_t<4>()(idata, kdata, odata); break;
                        case 5:         conv2d_madi_t<5>()(idata, kdata, odata); break;
                        case 6:         conv2d_madi_t<6>()(idata, kdata, odata); break;
                        case 7:         conv2d_madi_t<7>()(idata, kdata, odata); break;
                        case 8:         conv2d_madi_t<8>()(idata, kdata, odata); break;
                        case 9:         conv2d_madi_t<9>()(idata, kdata, odata); break;
                        case 10:        conv2d_madi_t<10>()(idata, kdata, odata); break;
                        case 11:        conv2d_madi_t<11>()(idata, kdata, odata); break;
                        case 12:        conv2d_madi_t<12>()(idata, kdata, odata); break;
                        case 13:        conv2d_madi_t<13>()(idata, kdata, odata); break;
                        case 14:        conv2d_madi_t<14>()(idata, kdata, odata); break;
                        case 15:        conv2d_madi_t<15>()(idata, kdata, odata); break;
                        default:        conv2d_mad_t()(idata, kdata, odata); break;
                        }
                }
        };
}

