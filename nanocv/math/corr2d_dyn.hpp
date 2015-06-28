#pragma once

#include "corr2d_mdk.hpp"
#include "corr2d_mdo.hpp"

namespace ncv
{
        namespace math
        {
                ///
                /// \brief 2D correlation: idata += odata @ kdata (by decoding the kernel size at runtime)
                ///
                template
                <
                        typename tmatrixo,
                        typename tmatrixk = tmatrixo,
                        typename tmatrixi = tmatrixo,
                        typename tscalar = typename tmatrixi::Scalar
                >
                void corr2d_dyn(const tmatrixo& odata, const tmatrixk& kdata, tmatrixi& idata)
                {
                        assert(idata.rows() + 1 == kdata.rows() + odata.rows());
                        assert(idata.cols() + 1 == kdata.cols() + odata.cols());

                        const auto kcols = kdata.cols();

                        // decode at run-time the kernel size
                        switch (kcols)
                        {
                        case 1:         corr2d_mdki<1>(odata, kdata, idata); break;
                        case 2:         corr2d_mdki<2>(odata, kdata, idata); break;
                        case 3:         corr2d_mdki<3>(odata, kdata, idata); break;
                        case 4:         corr2d_mdki<4>(odata, kdata, idata); break;
                        case 5:         corr2d_mdki<5>(odata, kdata, idata); break;
                        case 6:         corr2d_mdki<6>(odata, kdata, idata); break;
                        case 7:         corr2d_mdki<7>(odata, kdata, idata); break;
                        case 8:         corr2d_mdki<8>(odata, kdata, idata); break;
                        case 9:         corr2d_mdki<9>(odata, kdata, idata); break;
                        case 10:        corr2d_mdki<10>(odata, kdata, idata); break;
                        case 11:        corr2d_mdki<11>(odata, kdata, idata); break;
                        case 12:        corr2d_mdki<12>(odata, kdata, idata); break;
                        case 13:        corr2d_mdki<13>(odata, kdata, idata); break;
                        case 14:        corr2d_mdki<14>(odata, kdata, idata); break;
                        case 15:        corr2d_mdki<15>(odata, kdata, idata); break;
                        default:        corr2d_mdk(odata, kdata, idata); break;
                        }
                }
        }
}

