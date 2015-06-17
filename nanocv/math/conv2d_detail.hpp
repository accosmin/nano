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
                        typename tscalar = typename tmatrixi::Scalar
                >
                void conv_cpp(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata)
                {
                        const auto orows = odata.rows();
                        const auto ocols = odata.cols();
                        const auto krows = kdata.rows();
                        const auto kcols = kdata.cols();
                        const auto icols = idata.cols();

                        for (auto r = 0; r < orows; r ++)
                        {
                                tscalar* podata = odata.data() + r * ocols;

                                for (auto kr = 0; kr < krows; kr ++)
                                {
                                        const tscalar* pidata = idata.data() + (r + kr) * icols;
                                        const tscalar* pkdata = kdata.data() + kr * kcols;

                                        for (auto c = 0; c < ocols; c ++)
                                        {
                                                tscalar sum = 0;
                                                for (auto kc = 0; kc < kcols; kc ++)
                                                {
                                                        sum += pidata[c + kc] * pkdata[kc];
                                                }
                                                podata[c] += sum;
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
                void conv_dot(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata)
                {
                        const auto orows = odata.rows();
                        const auto ocols = odata.cols();
                        const auto krows = kdata.rows();
                        const auto kcols = kdata.cols();
                        const auto icols = idata.cols();

                        for (auto r = 0; r < orows; r ++)
                        {
                                tscalar* podata = odata.data() + r * ocols;

                                for (auto kr = 0; kr < krows; kr ++)
                                {
                                        const tscalar* pidata = idata.data() + (r + kr) * icols;
                                        const tscalar* pkdata = kdata.data() + kr * kcols;

                                        for (auto c = 0; c < ocols; c ++)
                                        {
                                                podata[c] += math::dot<tscalar>(pidata + c, pkdata, kcols);
                                        }
                                }
                        }
                }

                template
                <
                        int kcols,
                        typename tmatrixi,
                        typename tmatrixk = tmatrixi,
                        typename tmatrixo = tmatrixi,
                        typename tscalar = typename tmatrixi::Scalar
                >
                void conv_doti(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata)
                {
                        const auto orows = odata.rows();
                        const auto ocols = odata.cols();
                        const auto krows = kdata.rows();
                        const auto icols = idata.cols();

                        for (auto r = 0; r < orows; r ++)
                        {
                                tscalar* podata = odata.data() + r * ocols;

                                for (auto kr = 0; kr < krows; kr ++)
                                {
                                        const tscalar* pidata = idata.data() + (r + kr) * icols;
                                        const tscalar* pkdata = kdata.data() + kr * kcols;

                                        for (auto c = 0; c < ocols; c ++)
                                        {
                                                podata[c] += math::dot<kcols>(pidata + c, pkdata);
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
                void conv_mad(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata)
                {
                        const auto orows = odata.rows();
                        const auto ocols = odata.cols();
                        const auto krows = kdata.rows();
                        const auto kcols = kdata.cols();
                        const auto icols = idata.cols();

                        for (auto r = 0; r < orows; r ++)
                        {
                                tscalar* podata = odata.data() + r * ocols;

                                for (auto kr = 0; kr < krows; kr ++)
                                {
                                        const tscalar* pidata = idata.data() + (r + kr) * icols;
                                        const tscalar* pkdata = kdata.data() + kr * kcols;

                                        for (auto kc = 0; kc < kcols; kc ++)
                                        {
                                                math::mad<tscalar>(pidata + kc, pkdata[kc], ocols, podata);
                                        }
                                }
                        }
                }

                template
                <
                        int ocols,
                        typename tmatrixi,
                        typename tmatrixk = tmatrixi,
                        typename tmatrixo = tmatrixi,
                        typename tscalar = typename tmatrixi::Scalar
                >
                void conv_madi(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata)
                {
                        const auto orows = odata.rows();
                        const auto krows = kdata.rows();
                        const auto kcols = kdata.cols();
                        const auto icols = idata.cols();

                        for (auto r = 0; r < orows; r ++)
                        {
                                tscalar* podata = odata.data() + r * ocols;

                                for (auto kr = 0; kr < krows; kr ++)
                                {
                                        const tscalar* pidata = idata.data() + (r + kr) * icols;
                                        const tscalar* pkdata = kdata.data() + kr * kcols;

                                        for (auto kc = 0; kc < kcols; kc ++)
                                        {
                                                math::mad<ocols>(pidata + kc, pkdata[kc], podata);
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
                void conv_dyn(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata)
                {
                        const auto kcols = kdata.cols();
                        const auto ocols = odata.cols();                        

                        // decode at run-time the kernel size
                        if (kcols < ocols)
                        {
                                switch (kcols)
                                {
                                case 1:         conv_doti<1>(idata, kdata, odata); break;
                                case 2:         conv_doti<2>(idata, kdata, odata); break;
                                case 3:         conv_doti<3>(idata, kdata, odata); break;
                                case 4:         conv_doti<4>(idata, kdata, odata); break;
                                case 5:         conv_doti<5>(idata, kdata, odata); break;
                                case 6:         conv_doti<6>(idata, kdata, odata); break;
                                case 7:         conv_doti<7>(idata, kdata, odata); break;
                                case 8:         conv_doti<8>(idata, kdata, odata); break;
                                case 9:         conv_doti<9>(idata, kdata, odata); break;
                                case 10:        conv_doti<10>(idata, kdata, odata); break;
                                case 11:        conv_doti<11>(idata, kdata, odata); break;
                                case 12:        conv_doti<12>(idata, kdata, odata); break;
                                case 13:        conv_doti<13>(idata, kdata, odata); break;
                                case 14:        conv_doti<14>(idata, kdata, odata); break;
                                case 15:        conv_doti<15>(idata, kdata, odata); break;
                                default:        conv_dot(idata, kdata, odata); break;
                                }
                        }
                        else
                        {
                                switch (ocols)
                                {
                                case 1:         conv_madi<1>(idata, kdata, odata); break;
                                case 2:         conv_madi<2>(idata, kdata, odata); break;
                                case 3:         conv_madi<3>(idata, kdata, odata); break;
                                case 4:         conv_madi<4>(idata, kdata, odata); break;
                                case 5:         conv_madi<5>(idata, kdata, odata); break;
                                case 6:         conv_madi<6>(idata, kdata, odata); break;
                                case 7:         conv_madi<7>(idata, kdata, odata); break;
                                case 8:         conv_madi<8>(idata, kdata, odata); break;
                                case 9:         conv_madi<9>(idata, kdata, odata); break;
                                case 10:        conv_madi<10>(idata, kdata, odata); break;
                                case 11:        conv_madi<11>(idata, kdata, odata); break;
                                case 12:        conv_madi<12>(idata, kdata, odata); break;
                                case 13:        conv_madi<13>(idata, kdata, odata); break;
                                case 14:        conv_madi<14>(idata, kdata, odata); break;
                                case 15:        conv_madi<15>(idata, kdata, odata); break;
                                default:        conv_mad(idata, kdata, odata); break;
                                }
                        }
                }
        }
}

