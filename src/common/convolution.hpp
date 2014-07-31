#ifndef NANOCV_CONVOLUTION_H
#define NANOCV_CONVOLUTION_H

#include <functional>
#include <cassert>
#include "dot.hpp"

namespace ncv
{
        ///
        /// 2D convolution: odata += idata @ kdata (using Eigen 2D blocks)
        ///
        template
        <
                typename tmatrixi,
                typename tmatrixk = tmatrixi,
                typename tmatrixo = tmatrixi,
                typename tscalar = typename tmatrixi::Scalar
        >
        void conv_eig_add(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata)
        {
                for (auto r = 0; r < odata.rows(); r ++)
                {
                        for (auto c = 0; c < odata.cols(); c ++)
                        {
                                odata(r, c) +=
                                kdata.cwiseProduct(idata.block(r, c, kdata.rows(), kdata.cols())).sum();
                        }
                }
        }

        ///
        /// 2D convolution: odata = idata @ kdata (using Eigen 2D blocks)
        ///
        template
        <
                typename tmatrixi,
                typename tmatrixk = tmatrixi,
                typename tmatrixo = tmatrixi,
                typename tscalar = typename tmatrixi::Scalar
        >
        void conv_eig_set(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata)
        {
                for (auto r = 0; r < odata.rows(); r ++)
                {
                        for (auto c = 0; c < odata.cols(); c ++)
                        {
                                odata(r, c) =
                                kdata.cwiseProduct(idata.block(r, c, kdata.rows(), kdata.cols())).sum();
                        }
                }
        }

        ///
        /// outer 2D convolution: odata += idata @ kdata (using Eigen 2D blocks)
        ///
        template
        <
                typename tmatrixi,
                typename tmatrixk = tmatrixi,
                typename tmatrixo = tmatrixi,
                typename tscalar = typename tmatrixi::Scalar
        >
        void outer_conv_eig_add(const tmatrixk& kdata, const tmatrixo& odata, tmatrixi& idata)
        {
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
                        bool tsum,
                        typename tmatrixi,
                        typename tmatrixk = tmatrixi,
                        typename tmatrixo = tmatrixi,
                        typename tdotop,
                        typename tscalar = typename tmatrixi::Scalar
                >
                void conv_dot(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata, tdotop dotop)
                {
                        const auto orows = odata.rows();
                        const auto ocols = odata.cols();
                        const auto krows = kdata.rows();
                        const auto kcols = kdata.cols();

                        const auto icols = ocols + kcols - 1;

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
                                                const tscalar sum = dotop(ppidata + c, ppkdata);

                                                if (!tsum && kr == 0)
                                                {
                                                        ppodata[c] = sum;
                                                }
                                                else
                                                {
                                                        ppodata[c] += sum;
                                                }
                                        }
                                }
                        }
                }

                template
                <
                        bool tsum,
                        typename tmatrixi,
                        typename tmatrixk = tmatrixi,
                        typename tmatrixo = tmatrixi,
                        typename tscalar = typename tmatrixi::Scalar
                >
                void conv_dot(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata)
                {
                        const auto kcols = kdata.cols();

                        using std::placeholders::_1;
                        using std::placeholders::_2;

                        // decode at run-time the kernel size
                        if (kcols == 1) { conv_dot<tsum>(idata, kdata, odata, dot<tscalar, 1>); }
                        else if (kcols == 2) { conv_dot<tsum>(idata, kdata, odata, dot<tscalar, 2>); }
                        else if (kcols == 3) { conv_dot<tsum>(idata, kdata, odata, dot<tscalar, 3>); }
                        else if (kcols == 4) { conv_dot<tsum>(idata, kdata, odata, dot<tscalar, 4>); }
                        else if (kcols == 5) { conv_dot<tsum>(idata, kdata, odata, dot<tscalar, 5>); }
                        else if (kcols == 6) { conv_dot<tsum>(idata, kdata, odata, dot<tscalar, 6>); }
                        else if (kcols == 7) { conv_dot<tsum>(idata, kdata, odata, dot<tscalar, 7>); }
                        else if (kcols == 8) { conv_dot<tsum>(idata, kdata, odata, dot<tscalar, 8>); }
                        else if (kcols == 9) { conv_dot<tsum>(idata, kdata, odata, dot<tscalar, 9>); }
                        else if (kcols == 10) { conv_dot<tsum>(idata, kdata, odata, dot<tscalar, 10>); }
                        else if (kcols == 11) { conv_dot<tsum>(idata, kdata, odata, dot<tscalar, 11>); }
                        else if (kcols == 12) { conv_dot<tsum>(idata, kdata, odata, dot<tscalar, 12>); }
                        else if (kcols == 13) { conv_dot<tsum>(idata, kdata, odata, dot<tscalar, 13>); }
                        else if (kcols == 14) { conv_dot<tsum>(idata, kdata, odata, dot<tscalar, 14>); }
                        else if (kcols == 15) { conv_dot<tsum>(idata, kdata, odata, dot<tscalar, 15>); }
                        else
                        {
                                conv_dot<tsum>(idata, kdata, odata, 
                                               std::bind(dot<tscalar, decltype(kcols)>, _1, _2, kcols));
                        }
                }
        }

        ///
        /// 2D convolution: odata += idata @ kdata (using a dot operator)
        ///
        template
        <
                typename tmatrixi,
                typename tmatrixk = tmatrixi,
                typename tmatrixo = tmatrixi,
                typename tscalar = typename tmatrixi::Scalar
        >
        void conv_dot_add(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata)
        {
                detail::conv_dot<true>(idata, kdata, odata);
        }

        ///
        /// 2D convolution: odata = idata @ kdata (using a dot operator)
        ///
        template
        <
                typename tmatrixi,
                typename tmatrixk = tmatrixi,
                typename tmatrixo = tmatrixi,
                typename tscalar = typename tmatrixi::Scalar
        >
        void conv_dot_set(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata)
        {
                detail::conv_dot<false>(idata, kdata, odata);
        }

        ///
        /// 2D convolution for compile-time kernel size: odata += idata @ kdata (using a dot operator)
        ///
        template
        <
                int tsize,
                typename tmatrixi,
                typename tmatrixk = tmatrixi,
                typename tmatrixo = tmatrixi,
                typename tscalar = typename tmatrixi::Scalar
        >
        void conv_dot_add(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata)
        {
                assert(tsize == kdata.cols());

                detail::conv_dot<true>(idata, kdata, odata, dot<tscalar, tsize>);
        }

        ///
        /// 2D convolution for compile-time kernel size: odata = idata @ kdata (using a dot operator)
        ///
        template
        <
                int tsize,
                typename tmatrixi,
                typename tmatrixk = tmatrixi,
                typename tmatrixo = tmatrixi,
                typename tscalar = typename tmatrixi::Scalar
        >
        void conv_dot_set(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata)
        {
                assert(tsize == kdata.cols());

                detail::conv_dot<false>(idata, kdata, odata, dot<tscalar, tsize>);
        }
        
        ///
        /// outer 2D convolution: odata += idata @ kdata (using a dot product)
        ///
        template
        <
                typename tmatrixi,
                typename tmatrixk = tmatrixi,
                typename tmatrixo = tmatrixi,
                typename tscalar = typename tmatrixi::Scalar
        >
        void outer_conv_dot_add(const tmatrixk& kdata, const tmatrixo& odata, tmatrixi& idata)
        {
                for (auto r = 0; r < odata.rows(); r ++)
                {
                        for (auto c = 0; c < odata.cols(); c ++)
                        {
                                idata.block(r, c, kdata.rows(), kdata.cols()) += kdata * odata(r, c);
                        }
                }
        }
}

#endif // NANOCV_CONVOLUTION_H

