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
                        typename tscalar = typename tmatrixi::Scalar
                >
                void corr_madk(const tmatrixo& odata, const tmatrixk& kdata, tmatrixi& idata)
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
                                                math::mad<tscalar>(pkdata, podata[c], kcols, pidata + c);
                                        }
                                }
                        }
                }

                template
                <
                        int kcols,
                        typename tmatrixo,
                        typename tmatrixk = tmatrixo,
                        typename tmatrixi = tmatrixo,
                        typename tscalar = typename tmatrixi::Scalar
                >
                void corr_madki(const tmatrixo& odata, const tmatrixk& kdata, tmatrixi& idata)
                {
                        const auto orows = odata.rows();
                        const auto ocols = odata.cols();
                        const auto krows = kdata.rows();
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
                                                math::mad<kcols>(pkdata, podata[c], pidata + c);
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
                void corr_mado(const tmatrixo& odata, const tmatrixk& kdata, tmatrixi& idata)
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
                                                math::mad<tscalar>(podata, pkdata[kc], ocols, pidata + kc);
                                        }
                                }
                        }
                }

                template
                <
                        int ocols,
                        typename tmatrixo,
                        typename tmatrixk = tmatrixo,
                        typename tmatrixi = tmatrixo,
                        typename tscalar = typename tmatrixi::Scalar
                >
                void corr_madoi(const tmatrixo& odata, const tmatrixk& kdata, tmatrixi& idata)
                {
                        const auto orows = odata.rows();
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
                                                math::mad<ocols>(podata, pkdata[kc], pidata + kc);
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
                        case 1:         corr_madki<1>(odata, kdata, idata); break;
                        case 2:         corr_madki<2>(odata, kdata, idata); break;
                        case 3:         corr_madki<3>(odata, kdata, idata); break;
                        case 4:         corr_madki<4>(odata, kdata, idata); break;
                        case 5:         corr_madki<5>(odata, kdata, idata); break;
                        case 6:         corr_madki<6>(odata, kdata, idata); break;
                        case 7:         corr_madki<7>(odata, kdata, idata); break;
                        case 8:         corr_madki<8>(odata, kdata, idata); break;
                        case 9:         corr_madki<9>(odata, kdata, idata); break;
                        case 10:        corr_madki<10>(odata, kdata, idata); break;
                        case 11:        corr_madki<11>(odata, kdata, idata); break;
                        case 12:        corr_madki<12>(odata, kdata, idata); break;
                        case 13:        corr_madki<13>(odata, kdata, idata); break;
                        case 14:        corr_madki<14>(odata, kdata, idata); break;
                        case 15:        corr_madki<15>(odata, kdata, idata); break;
                        default:        corr_madk(odata, kdata, idata); break;
                        }
                }
        }
}

