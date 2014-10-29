#pragma once

#include "common/dot.hpp"
#include "common/mad.hpp"
#include <cassert>

namespace ncv
{
        ///
        /// \brief 2D convolution: odata += idata @ kdata (using Eigen 2D blocks)
        ///
        template
        <
                typename tmatrixi,
                typename tmatrixk = tmatrixi,
                typename tmatrixo = tmatrixi,
                typename tscalar = typename tmatrixi::Scalar
        >
        void conv2d_eig(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata)
        {
                assert(idata.rows() + 1 == kdata.rows() + odata.rows());
                assert(idata.cols() + 1 == kdata.cols() + odata.cols());

                for (auto r = 0; r < odata.rows(); r ++)
                {
                        for (auto c = 0; c < odata.cols(); c ++)
                        {
                                odata(r, c) += kdata.cwiseProduct(idata.block(r, c, kdata.rows(), kdata.cols())).sum();
                        }
                }
        }

        namespace detail
        {
                template
                <
                        typename tmatrixi,
                        typename tmatrixk = tmatrixi,
                        typename tmatrixo = tmatrixi,
                        typename tscalar = typename tmatrixi::Scalar
                >
                static void conv_cpp(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata)
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
                                                tscalar sum = 0;
                                                for (auto kc = 0; kc < kcols; kc ++)
                                                {
                                                        sum += ppidata[c + kc] * ppkdata[kc];
                                                }
                                                ppodata[c] += sum;
                                        }
                                }
                        }
                }

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
                static void conv_dyn(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata)
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
                                default:        conv_dot(idata, kdata, odata, dot<tscalar>); break;
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
                                default:        conv_mad(idata, kdata, odata, mad<tscalar>); break;
                                }
                        }
                }
        }

        ///
        /// \brief 2D convolution: odata += idata @ kdata (using plain array indexing)
        ///
        template
        <
                typename tmatrixi,
                typename tmatrixk = tmatrixi,
                typename tmatrixo = tmatrixi,
                typename tscalar = typename tmatrixi::Scalar
        >
        void conv2d_cpp(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata)
        {
                assert(idata.rows() + 1 == kdata.rows() + odata.rows());
                assert(idata.cols() + 1 == kdata.cols() + odata.cols());

                detail::conv_cpp(idata, kdata, odata);
        }

        ///
        /// \brief 2D convolution: odata += idata @ kdata (using a dot operator)
        ///
        template
        <
                typename tmatrixi,
                typename tmatrixk = tmatrixi,
                typename tmatrixo = tmatrixi,
                typename tscalar = typename tmatrixi::Scalar
        >
        void conv2d_dot(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata)
        {
                assert(idata.rows() + 1 == kdata.rows() + odata.rows());
                assert(idata.cols() + 1 == kdata.cols() + odata.cols());

                detail::conv_dot(idata, kdata, odata, dot<tscalar>);
        }

        ///
        /// \brief 2D convolution: odata += idata @ kdata (using a mad operator)
        ///
        template
        <
                typename tmatrixi,
                typename tmatrixk = tmatrixi,
                typename tmatrixo = tmatrixi,
                typename tscalar = typename tmatrixi::Scalar
        >
        void conv2d_mad(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata)
        {
                assert(idata.rows() + 1 == kdata.rows() + odata.rows());
                assert(idata.cols() + 1 == kdata.cols() + odata.cols());

                detail::conv_mad(idata, kdata, odata, mad<tscalar>);
        }

        ///
        /// \brief 2D convolution: odata += idata @ kdata (by decoding the kernel size at runtime)
        ///
        template
        <
                typename tmatrixi,
                typename tmatrixk = tmatrixi,
                typename tmatrixo = tmatrixi,
                typename tscalar = typename tmatrixi::Scalar
        >
        void conv2d_dyn(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata)
        {
                assert(idata.rows() + 1 == kdata.rows() + odata.rows());
                assert(idata.cols() + 1 == kdata.cols() + odata.cols());
                assert(tsize == kdata.cols());

                detail::conv_dyn(idata, kdata, odata);
        }
}

