#pragma once

#include "mad.hpp"
#include <cassert>

namespace ncv
{
        namespace detail
        {
                template
                <
                        typename tmatrixo,
                        typename tmatrixk = tmatrixo,
                        typename tmatrixi = tmatrixo,
                        typename tscalar = typename tmatrixi::Scalar
                >
                static void corr_cpp(const tmatrixo& odata, const tmatrixk& kdata, tmatrixi& idata)
                {
                        const auto orows = odata.rows();
                        const auto ocols = odata.cols();
                        const auto krows = kdata.rows();
                        const auto kcols = kdata.cols();
                        const auto icols = idata.cols();

                        const tscalar* pkdata = kdata.data();
                        const tscalar* podata = odata.data();
                        tscalar* pidata = idata.data();

                        for (auto r = 0; r < orows; r ++)
                        {
                                const tscalar* ppodata = podata + r * ocols;

                                for (auto kr = 0; kr < krows; kr ++)
                                {
                                        tscalar* ppidata = pidata + (r + kr) * icols;
                                        const tscalar* ppkdata = pkdata + kr * kcols;

                                        for (auto c = 0; c < ocols; c ++)
                                        {
                                                for (auto kc = 0; kc < kcols; kc ++)
                                                {
                                                        ppidata[c + kc] += ppkdata[kc] * ppodata[c];
                                                }
                                        }
                                }
                        }
                }

                template
                <
                        typename tmatrixo,
                        typename tmatrixk = tmatrixo,
                        typename tmatrixi = tmatrixo,
                        typename tmadop,
                        typename tscalar = typename tmatrixi::Scalar
                >
                static void corr_mad(const tmatrixo& odata, const tmatrixk& kdata, tmatrixi& idata, const tmadop& madop)
                {
                        const auto orows = odata.rows();
                        const auto ocols = odata.cols();
                        const auto krows = kdata.rows();
                        const auto kcols = kdata.cols();
                        const auto icols = idata.cols();

                        const tscalar* pkdata = kdata.data();
                        const tscalar* podata = odata.data();
                        tscalar* pidata = idata.data();

                        for (auto r = 0; r < orows; r ++)
                        {
                                const tscalar* ppodata = podata + r * ocols;

                                for (auto kr = 0; kr < krows; kr ++)
                                {
                                        tscalar* ppidata = pidata + (r + kr) * icols;
                                        const tscalar* ppkdata = pkdata + kr * kcols;

                                        for (auto c = 0; c < ocols; c ++)
                                        {
                                                madop(ppkdata, ppodata[c], kcols, ppidata + c);
                                        }
                                }
                        }
                }

                template
                <
                        typename tmatrixo,
                        typename tmatrixk = tmatrixo,
                        typename tmatrixi = tmatrixo,
                        typename tscalar = typename tmatrixi::Scalar
                >
                static void corr_dyn(const tmatrixo& odata, const tmatrixk& kdata, tmatrixi& idata)
                {
                        const auto kcols = kdata.cols();

                        // decode at run-time the kernel size
                        switch (kcols)
                        {
                        case 1:         corr_mad(odata, kdata, idata, mad<tscalar, 1>); break;
                        case 2:         corr_mad(odata, kdata, idata, mad<tscalar, 2>); break;
                        case 3:         corr_mad(odata, kdata, idata, mad<tscalar, 3>); break;
                        case 4:         corr_mad(odata, kdata, idata, mad<tscalar, 4>); break;
                        case 5:         corr_mad(odata, kdata, idata, mad<tscalar, 5>); break;
                        case 6:         corr_mad(odata, kdata, idata, mad<tscalar, 6>); break;
                        case 7:         corr_mad(odata, kdata, idata, mad<tscalar, 7>); break;
                        case 8:         corr_mad(odata, kdata, idata, mad<tscalar, 8>); break;
                        case 9:         corr_mad(odata, kdata, idata, mad<tscalar, 9>); break;
                        case 10:        corr_mad(odata, kdata, idata, mad<tscalar, 10>); break;
                        case 11:        corr_mad(odata, kdata, idata, mad<tscalar, 11>); break;
                        case 12:        corr_mad(odata, kdata, idata, mad<tscalar, 12>); break;
                        case 13:        corr_mad(odata, kdata, idata, mad<tscalar, 13>); break;
                        case 14:        corr_mad(odata, kdata, idata, mad<tscalar, 14>); break;
                        case 15:        corr_mad(odata, kdata, idata, mad<tscalar, 15>); break;
                        default:        corr_mad(odata, kdata, idata, mad<tscalar>); break;
                        }
                }
        }
}

