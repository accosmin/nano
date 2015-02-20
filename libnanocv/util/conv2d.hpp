#pragma once

#include "conv2d_detail.hpp"
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

                detail::conv_dyn(idata, kdata, odata);
        }

        ///
        /// \brief 2D convolution: odata += idata @ kdata (using a Toeplitz matrix)
        ///
        template
        <
                typename tmatrixi,
                typename tmatrixk = tmatrixi,
                typename tmatrixo = tmatrixi,
                typename tscalar = typename tmatrixi::Scalar
        >
        void conv2d_toe(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata)
        {
                assert(idata.rows() + 1 == kdata.rows() + odata.rows());
                assert(idata.cols() + 1 == kdata.cols() + odata.cols());

                detail::conv_toeplitz(idata, kdata, odata);
        }
}

