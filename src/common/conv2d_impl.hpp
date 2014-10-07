#pragma once

#include "dot.hpp"
#include "mad.hpp"

namespace ncv
{
        namespace detail
        {
                template
                <
                        typename tmatrixi,
                        typename tmatrixk = tmatrixi,
                        typename tmatrixo = tmatrixi,
                        typename tdotop,
                        typename tscalar = typename tmatrixi::Scalar
                >
                static void conv_dot(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata, const tdotop& dotop)
                {
                        const auto orows = odata.rows();
                        const auto ocols = odata.cols();
                        const auto krows = kdata.rows();
                        const auto kcols = kdata.cols();
                        const auto icols = idata.cols();

                        const tscalar* pidata = idata.data();
                        const tscalar* pkdata = kdata.data();
                        tscalar* podata = odata.data();

                        for (auto r = 0; r < orows; r ++)
                        {
                                tscalar* ppodata = podata + r * ocols;

                                for (auto kr = 0; kr < krows; kr ++)
                                {
                                        const tscalar* ppidata = pidata + (r + kr) * icols;
                                        const tscalar* ppkdata = pkdata + kr * kcols;

                                        for (auto c = 0; c < ocols; c ++)
                                        {
                                                ppodata[c] += dotop(ppidata + c, ppkdata, kcols);
                                        }
                                }
                        }
                }

                template
                <
                        typename tmatrixi,
                        typename tmatrixk = tmatrixi,
                        typename tmatrixo = tmatrixi,
                        typename tmadop,
                        typename tscalar = typename tmatrixi::Scalar
                >
                static void conv_mad(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata, const tmadop& madop)
                {
                        const auto orows = odata.rows();
                        const auto ocols = odata.cols();
                        const auto krows = kdata.rows();
                        const auto kcols = kdata.cols();
                        const auto icols = idata.cols();

                        const tscalar* pidata = idata.data();
                        const tscalar* pkdata = kdata.data();
                        tscalar* podata = odata.data();

                        for (auto r = 0; r < orows; r ++)
                        {
                                tscalar* ppodata = podata + r * ocols;

                                for (auto kr = 0; kr < krows; kr ++)
                                {
                                        const tscalar* ppidata = pidata + (r + kr) * icols;
                                        const tscalar* ppkdata = pkdata + kr * kcols;

                                        for (auto kc = 0; kc < kcols; kc ++)
                                        {
                                                madop(ppidata + kc, ppkdata[kc], ocols, ppodata);
                                        }
                                }
                        }
                }

                template
                <
                        typename tmatrixi,
                        typename tmatrixk = tmatrixi,
                        typename tmatrixo = tmatrixi,
                        typename tscalar = typename tmatrixi::Scalar
                >
                static void conv_dot(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata)
                {
                        const auto kcols = kdata.cols();
                        const auto ocols = odata.cols();

                        // decode at run-time the kernel size
                        if (kcols < ocols)
                        {
                                switch (kcols)
                                {
                                case 1:         conv_dot(idata, kdata, odata, dot<tscalar, 1>); break;
                                case 2:         conv_dot(idata, kdata, odata, dot<tscalar, 2>); break;
                                case 3:         conv_dot(idata, kdata, odata, dot<tscalar, 3>); break;
                                case 4:         conv_dot(idata, kdata, odata, dot<tscalar, 4>); break;
                                case 5:         conv_dot(idata, kdata, odata, dot<tscalar, 5>); break;
                                case 6:         conv_dot(idata, kdata, odata, dot<tscalar, 6>); break;
                                case 7:         conv_dot(idata, kdata, odata, dot<tscalar, 7>); break;
                                case 8:         conv_dot(idata, kdata, odata, dot<tscalar, 8>); break;
                                case 9:         conv_dot(idata, kdata, odata, dot<tscalar, 9>); break;
                                case 10:        conv_dot(idata, kdata, odata, dot<tscalar, 10>); break;
                                case 11:        conv_dot(idata, kdata, odata, dot<tscalar, 11>); break;
                                case 12:        conv_dot(idata, kdata, odata, dot<tscalar, 12>); break;
                                case 13:        conv_dot(idata, kdata, odata, dot<tscalar, 13>); break;
                                case 14:        conv_dot(idata, kdata, odata, dot<tscalar, 14>); break;
                                case 15:        conv_dot(idata, kdata, odata, dot<tscalar, 15>); break;
                                default:        conv_dot(idata, kdata, odata, dot<tscalar, decltype(kcols)>); break;
                                }
                        }
                        else
                        {
                                switch (ocols)
                                {
                                case 1:         conv_mad(idata, kdata, odata, mad<tscalar, 1>); break;
                                case 2:         conv_mad(idata, kdata, odata, mad<tscalar, 2>); break;
                                case 3:         conv_mad(idata, kdata, odata, mad<tscalar, 3>); break;
                                case 4:         conv_mad(idata, kdata, odata, mad<tscalar, 4>); break;
                                case 5:         conv_mad(idata, kdata, odata, mad<tscalar, 5>); break;
                                case 6:         conv_mad(idata, kdata, odata, mad<tscalar, 6>); break;
                                case 7:         conv_mad(idata, kdata, odata, mad<tscalar, 7>); break;
                                case 8:         conv_mad(idata, kdata, odata, mad<tscalar, 8>); break;
                                case 9:         conv_mad(idata, kdata, odata, mad<tscalar, 9>); break;
                                case 10:        conv_mad(idata, kdata, odata, mad<tscalar, 10>); break;
                                case 11:        conv_mad(idata, kdata, odata, mad<tscalar, 11>); break;
                                case 12:        conv_mad(idata, kdata, odata, mad<tscalar, 12>); break;
                                case 13:        conv_mad(idata, kdata, odata, mad<tscalar, 13>); break;
                                case 14:        conv_mad(idata, kdata, odata, mad<tscalar, 14>); break;
                                case 15:        conv_mad(idata, kdata, odata, mad<tscalar, 15>); break;
                                default:        conv_mad(idata, kdata, odata, mad<tscalar, decltype(ocols)>); break;
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
                static void iconv_mad(const tmatrixo& odata, const tmatrixk& kdata, tmatrixi& idata, const tmadop& madop)
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
                static void iconv_mad(const tmatrixo& odata, const tmatrixk& kdata, tmatrixi& idata)
                {
                        const auto kcols = kdata.cols();

                        // decode at run-time the kernel size
                        switch (kcols)
                        {
                        case 1:         iconv_mad(odata, kdata, idata, mad<tscalar, 1>); break;
                        case 2:         iconv_mad(odata, kdata, idata, mad<tscalar, 2>); break;
                        case 3:         iconv_mad(odata, kdata, idata, mad<tscalar, 3>); break;
                        case 4:         iconv_mad(odata, kdata, idata, mad<tscalar, 4>); break;
                        case 5:         iconv_mad(odata, kdata, idata, mad<tscalar, 5>); break;
                        case 6:         iconv_mad(odata, kdata, idata, mad<tscalar, 6>); break;
                        case 7:         iconv_mad(odata, kdata, idata, mad<tscalar, 7>); break;
                        case 8:         iconv_mad(odata, kdata, idata, mad<tscalar, 8>); break;
                        case 9:         iconv_mad(odata, kdata, idata, mad<tscalar, 9>); break;
                        case 10:        iconv_mad(odata, kdata, idata, mad<tscalar, 10>); break;
                        case 11:        iconv_mad(odata, kdata, idata, mad<tscalar, 11>); break;
                        case 12:        iconv_mad(odata, kdata, idata, mad<tscalar, 12>); break;
                        case 13:        iconv_mad(odata, kdata, idata, mad<tscalar, 13>); break;
                        case 14:        iconv_mad(odata, kdata, idata, mad<tscalar, 14>); break;
                        case 15:        iconv_mad(odata, kdata, idata, mad<tscalar, 15>); break;
                        default:        iconv_mad(odata, kdata, idata, mad<tscalar, decltype(kcols)>); break;
                        }
                }
        }
}

