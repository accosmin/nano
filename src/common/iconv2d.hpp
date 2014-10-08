#pragma once

#include "common/mad.hpp"
#include <cassert>

namespace ncv
{
        ///
        /// \brief inverse 2D convolution: idata += odata @ kdata (using Eigen 2D blocks)
        ///
        template
        <
                typename tmatrixo,
                typename tmatrixk = tmatrixo,
                typename tmatrixi = tmatrixo,
                typename tscalar = typename tmatrixi::Scalar
        >
        void iconv2d_eig(const tmatrixo& odata, const tmatrixk& kdata, tmatrixi& idata)
        {
                assert(idata.rows() + 1 == kdata.rows() + odata.rows());
                assert(idata.cols() + 1 == kdata.cols() + odata.cols());

                for (auto r = 0; r < odata.rows(); r ++)
                {
                        for (auto c = 0; c < odata.cols(); c ++)
                        {
                                idata.block(r, c, kdata.rows(), kdata.cols()) += kdata * odata(r, c);
                        }
                }
        }

        namespace detail
        {
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
        
        ///
        /// \brief inverse 2D convolution: idata += odata @ kdata (using a mad product)
        ///
        template
        <
                typename tmatrixo,
                typename tmatrixk = tmatrixo,
                typename tmatrixi = tmatrixo,
                typename tscalar = typename tmatrixi::Scalar
        >
        void iconv2d_mad(const tmatrixo& odata, const tmatrixk& kdata, tmatrixi& idata)
        {
                assert(idata.rows() + 1 == kdata.rows() + odata.rows());
                assert(idata.cols() + 1 == kdata.cols() + odata.cols());

                detail::iconv_mad(odata, kdata, idata);
        }

        ///
        /// \brief inverse 2D convolution for compile-time kernel size: idata += odata @ kdata (using a mad product)
        ///
        template
        <
                int tsize,
                typename tmatrixo,
                typename tmatrixk = tmatrixo,
                typename tmatrixi = tmatrixo,
                typename tscalar = typename tmatrixi::Scalar
        >
        void iconv2d_mad(const tmatrixo& odata, const tmatrixk& kdata, tmatrixi& idata)
        {
                assert(idata.rows() + 1 == kdata.rows() + odata.rows());
                assert(idata.cols() + 1 == kdata.cols() + odata.cols());

                detail::iconv_mad(odata, kdata, idata, mad<tscalar, tsize>);
        }
}

