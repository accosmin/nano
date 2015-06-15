#pragma once

#include "mad.hpp"

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
                void corr_cpp(const tmatrixo& odata, const tmatrixk& kdata, tmatrixi& idata)
                {
                        const auto orows = odata.rows();
                        const auto ocols = odata.cols();
                        const auto krows = kdata.rows();
                        const auto kcols = kdata.cols();
                        const auto icols = idata.cols();

                        for (auto r = 0; r < orows; r ++)
                        {
                                const tscalar* podata = odata.data() + r * ocols;

                                for (auto kr = 0; kr < krows; kr ++)
                                {
                                        tscalar* pidata = idata.data() + (r + kr) * icols;
                                        const tscalar* pkdata = kdata.data() + kr * kcols;

                                        for (auto c = 0; c < ocols; c ++)
                                        {
                                                for (auto kc = 0; kc < kcols; kc ++)
                                                {
                                                        pidata[c + kc] += pkdata[kc] * podata[c];
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
                void corr_madk(const tmatrixo& odata, const tmatrixk& kdata, tmatrixi& idata, const tmadop& madop)
                {
                        const auto orows = odata.rows();
                        const auto ocols = odata.cols();
                        const auto krows = kdata.rows();
                        const auto kcols = kdata.cols();
                        const auto icols = idata.cols();

                        for (auto r = 0; r < orows; r ++)
                        {
                                const tscalar* podata = odata.data() + r * ocols;

                                for (auto kr = 0; kr < krows; kr ++)
                                {
                                        tscalar* pidata = idata.data() + (r + kr) * icols;
                                        const tscalar* pkdata = kdata.data() + kr * kcols;

                                        for (auto c = 0; c < ocols; c ++)
                                        {
                                                madop(pkdata, podata[c], kcols, pidata + c);
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
                void corr_mado(const tmatrixo& odata, const tmatrixk& kdata, tmatrixi& idata, const tmadop& madop)
                {
                        const auto orows = odata.rows();
                        const auto ocols = odata.cols();
                        const auto krows = kdata.rows();
                        const auto kcols = kdata.cols();
                        const auto icols = idata.cols();

                        for (auto r = 0; r < orows; r ++)
                        {
                                const tscalar* podata = odata.data() + r * ocols;

                                for (auto kr = 0; kr < krows; kr ++)
                                {
                                        tscalar* pidata = idata.data() + (r + kr) * icols;
                                        const tscalar* pkdata = kdata.data() + kr * kcols;

                                        for (auto kc = 0; kc < kcols; kc ++)
                                        {
                                                madop(podata, pkdata[kc], ocols, pidata + kc);
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
                void corr_dyn(const tmatrixo& odata, const tmatrixk& kdata, tmatrixi& idata)
                {
                        const auto kcols = kdata.cols();

                        // decode at run-time the kernel size
                        switch (kcols)
                        {
                        case 1:         corr_madk(odata, kdata, idata, math::mad<tscalar, 1>); break;
                        case 2:         corr_madk(odata, kdata, idata, math::mad<tscalar, 2>); break;
                        case 3:         corr_madk(odata, kdata, idata, math::mad<tscalar, 3>); break;
                        case 4:         corr_madk(odata, kdata, idata, math::mad<tscalar, 4>); break;
                        case 5:         corr_madk(odata, kdata, idata, math::mad<tscalar, 5>); break;
                        case 6:         corr_madk(odata, kdata, idata, math::mad<tscalar, 6>); break;
                        case 7:         corr_madk(odata, kdata, idata, math::mad<tscalar, 7>); break;
                        case 8:         corr_madk(odata, kdata, idata, math::mad<tscalar, 8>); break;
                        case 9:         corr_madk(odata, kdata, idata, math::mad<tscalar, 9>); break;
                        case 10:        corr_madk(odata, kdata, idata, math::mad<tscalar, 10>); break;
                        case 11:        corr_madk(odata, kdata, idata, math::mad<tscalar, 11>); break;
                        case 12:        corr_madk(odata, kdata, idata, math::mad<tscalar, 12>); break;
                        case 13:        corr_madk(odata, kdata, idata, math::mad<tscalar, 13>); break;
                        case 14:        corr_madk(odata, kdata, idata, math::mad<tscalar, 14>); break;
                        case 15:        corr_madk(odata, kdata, idata, math::mad<tscalar, 15>); break;
                        default:        corr_madk(odata, kdata, idata, math::mad<tscalar>); break;
                        }
                }
        }
}

