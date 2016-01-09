#pragma once

#include "corr2d_mdk.hpp"
#include "corr2d_mdki.hpp"
#include "corr2d_mdo.hpp"
#include "corr2d_mdoi.hpp"

namespace tensor
{
        ///
        /// \brief 2D correlation: idata += odata @ kdata (by decoding the kernel size at runtime)
        ///
        struct corr2d_dyn_t
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

                        const auto kcols = kdata.cols();

                        // decode at run-time the kernel size
                        switch (kcols)
                        {
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

