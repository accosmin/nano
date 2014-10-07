#pragma once

#include "conv2d_impl.hpp"
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

                detail::conv_dot(idata, kdata, odata);
        }

        ///
        /// \brief 2D convolution for compile-time kernel size: odata += idata @ kdata (using a dot operator)
        ///
        template
        <
                int tsize,
                typename tmatrixi,
                typename tmatrixk = tmatrixi,
                typename tmatrixo = tmatrixi,
                typename tscalar = typename tmatrixi::Scalar
        >
        void conv2d_dot(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata)
        {
                assert(idata.rows() + 1 == kdata.rows() + odata.rows());
                assert(idata.cols() + 1 == kdata.cols() + odata.cols());
                assert(tsize == kdata.cols());

                detail::conv_dot(idata, kdata, odata, dot<tscalar, tsize>);
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

