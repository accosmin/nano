#ifndef NANOCV_CONVOLUTION_H
#define NANOCV_CONVOLUTION_H

#include <functional>
#include "dot.hpp"

namespace ncv
{
        namespace math
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
                void outer_conv_eig_add(tmatrixi& idata, const tmatrixk& kdata, const tmatrixo& odata)
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
                        ///
                        /// \brief 2D convolution using a dot-operator
                        ///
                        template
                        <
                                bool tsum,
                                typename tmatrixi,
                                typename tmatrixk = tmatrixi,
                                typename tmatrixo = tmatrixi,
                                typename tdotop,
                                typename tscalar = typename tmatrixi::Scalar
                        >
                        void conv_test(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata, tdotop dotop)
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

                        ///
                        /// \brief 2D convolution with runtime decoding of the kernel size
                        ///
                        template
                        <
                                bool tsum,
                                typename tmatrixi,
                                typename tmatrixk = tmatrixi,
                                typename tmatrixo = tmatrixi,
                                typename tscalar = typename tmatrixi::Scalar
                        >
                        void conv_test(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata)
                        {
                                const auto kcols = kdata.cols();

                                using std::placeholders::_1;
                                using std::placeholders::_2;

                                if (kcols == 1) { conv_test<tsum>(idata, kdata, odata, dot<tscalar, 1>); }
                                else if (kcols == 2) { conv_test<tsum>(idata, kdata, odata, dot<tscalar, 2>); }
                                else if (kcols == 3) { conv_test<tsum>(idata, kdata, odata, dot<tscalar, 3>); }
                                else if (kcols == 4) { conv_test<tsum>(idata, kdata, odata, dot<tscalar, 4>); }
                                else if (kcols == 5) { conv_test<tsum>(idata, kdata, odata, dot<tscalar, 5>); }
                                else if (kcols == 6) { conv_test<tsum>(idata, kdata, odata, dot<tscalar, 6>); }
                                else if (kcols == 7) { conv_test<tsum>(idata, kdata, odata, dot<tscalar, 7>); }
                                else if (kcols == 8) { conv_test<tsum>(idata, kdata, odata, dot<tscalar, 8>); }
                                else if (kcols == 9) { conv_test<tsum>(idata, kdata, odata, dot<tscalar, 9>); }
                                else if (kcols == 10) { conv_test<tsum>(idata, kdata, odata, dot<tscalar, 10>); }
                                else if (kcols == 11) { conv_test<tsum>(idata, kdata, odata, dot<tscalar, 11>); }
                                else if (kcols == 12) { conv_test<tsum>(idata, kdata, odata, dot<tscalar, 12>); }
                                else
                                {
                                        conv_test<tsum>(idata, kdata, odata,
                                                        std::bind(dot<tscalar, decltype(kcols)>, _1, _2, kcols));
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
                void conv_sum_test(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata)
                {
                        detail::conv_test<true>(idata, kdata, odata);
                }

                template
                <
                        typename tmatrixi,
                        typename tmatrixk = tmatrixi,
                        typename tmatrixo = tmatrixi,
                        typename tscalar = typename tmatrixi::Scalar
                >
                void conv_test(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata)
                {
                        detail::conv_test<false>(idata, kdata, odata);
                }
        }
}

#endif // NANOCV_CONVOLUTION_H

