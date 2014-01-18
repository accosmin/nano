#ifndef NANOCV_CONVOLUTION_H
#define NANOCV_CONVOLUTION_H

#include <eigen3/Eigen/Core>
#include "dot.hpp"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // 2D convolution.
        /////////////////////////////////////////////////////////////////////////////////////////

        namespace math
        {
		namespace impl
		{
                        // odata += idata @ kdata (using a column-based dot operator)
	                template
        	        <
                                typename tdot,
	                        typename tmatrix,
	                        typename tscalar = typename tmatrix::Scalar
        	        >
                        void conv_dot(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata, tdot dotop)
	                {
                                const auto icols = idata.cols();
                                const auto krows = kdata.rows(), kcols = kdata.cols();
                                const auto orows = odata.rows(), ocols = odata.cols();

                                for (auto r = 0; r < orows; r ++)
                                {
                                        const tscalar* pi = &idata(r, 0);
                                        tscalar* po = &odata(r, 0);

                                        for (auto kr = 0; kr < krows; kr ++, pi += icols)
                                        {
                                                const tscalar* pk = &kdata(kr, 0);

                                                for (auto c = 0; c < ocols; c ++)
                                                {
                                                        po[c] += dotop(pi + c, pk, kcols);
                                                }
                                        }
                                }
                	}

                        // odata += weight * idata @ kdata (using a column-based dot operator)
	                template
        	        <
                                typename tdot,
	                        typename tmatrix,
        	                typename tscalar = typename tmatrix::Scalar
                	>
                        void wconv_dot(const tmatrix& idata, const tmatrix& kdata, tscalar weight, tmatrix& odata, tdot dotop)
        	        {
                                const auto icols = idata.cols();
                                const auto krows = kdata.rows(), kcols = kdata.cols();
                                const auto orows = odata.rows(), ocols = odata.cols();

                                for (auto r = 0; r < orows; r ++)
                                {
                                        const tscalar* pi = &idata(r, 0);
                                        tscalar* po = &odata(r, 0);

                                        for (auto kr = 0; kr < krows; kr ++, pi += icols)
                                        {
                                                const tscalar* pk = &kdata(kr, 0);

                                                for (auto c = 0; c < ocols; c ++)
                                                {
                                                        po[c] += weight * dotop(pi + c, pk, kcols);
                                                }
                                        }
                                }
                	}
		}

                // odata += idata @ kdata
                template
                <
                        typename tmatrix,
                        typename tscalar = typename tmatrix::Scalar
                >
                void conv_dot(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
                {
                        const auto kcols = kdata.cols();

                        if (kcols == 3) { impl::conv_dot(idata, kdata, odata, dot<3, tscalar>); }
                        else if (kcols == 4) { impl::conv_dot(idata, kdata, odata, dot<4, tscalar>); }
                        else if (kcols == 5) { impl::conv_dot(idata, kdata, odata, dot<5, tscalar>); }
                        else if (kcols == 6) { impl::conv_dot(idata, kdata, odata, dot<6, tscalar>); }
                        else if (kcols == 7) { impl::conv_dot(idata, kdata, odata, dot<7, tscalar>); }
                        else if (kcols == 8) { impl::conv_dot(idata, kdata, odata, dot<8, tscalar>); }
                        else if (kcols == 9) { impl::conv_dot(idata, kdata, odata, dot<9, tscalar>); }
                        else if (kcols == 10) { impl::conv_dot(idata, kdata, odata, dot<10, tscalar>); }
                        else if (kcols == 11) { impl::conv_dot(idata, kdata, odata, dot<11, tscalar>); }
                        else if (kcols == 12) { impl::conv_dot(idata, kdata, odata, dot<12, tscalar>); }
                        else if (kcols == 13) { impl::conv_dot(idata, kdata, odata, dot<13, tscalar>); }
                        else if (kcols == 14) { impl::conv_dot(idata, kdata, odata, dot<14, tscalar>); }
                        else if (kcols == 15) { impl::conv_dot(idata, kdata, odata, dot<15, tscalar>); }
                        else if ((kcols & 0x3) == 0) { impl::conv_dot(idata, kdata, odata, dot_mod4<tscalar>); }
                        else { impl::conv_dot(idata, kdata, odata, dot_mod4x<tscalar>); }
                }

                // odata += weight * idata @ kdata
                template
                <
                        typename tmatrix,
                        typename tscalar = typename tmatrix::Scalar
                >
                void wconv_dot(const tmatrix& idata, const tmatrix& kdata, tscalar weight, tmatrix& odata)
                {
                        const auto kcols = kdata.cols();

                        if (kcols == 3) { impl::wconv_dot(idata, kdata, weight, odata, dot<3, tscalar>); }
                        else if (kcols == 4) { impl::wconv_dot(idata, kdata, weight, odata, dot<4, tscalar>); }
                        else if (kcols == 5) { impl::wconv_dot(idata, kdata, weight, odata, dot<5, tscalar>); }
                        else if (kcols == 6) { impl::wconv_dot(idata, kdata, weight, odata, dot<6, tscalar>); }
                        else if (kcols == 7) { impl::wconv_dot(idata, kdata, weight, odata, dot<7, tscalar>); }
                        else if (kcols == 8) { impl::wconv_dot(idata, kdata, weight, odata, dot<8, tscalar>); }
                        else if (kcols == 9) { impl::wconv_dot(idata, kdata, weight, odata, dot<9, tscalar>); }
                        else if (kcols == 10) { impl::wconv_dot(idata, kdata, weight, odata, dot<10, tscalar>); }
                        else if (kcols == 11) { impl::wconv_dot(idata, kdata, weight, odata, dot<11, tscalar>); }
                        else if (kcols == 12) { impl::wconv_dot(idata, kdata, weight, odata, dot<12, tscalar>); }
                        else if (kcols == 13) { impl::wconv_dot(idata, kdata, weight, odata, dot<13, tscalar>); }
                        else if (kcols == 14) { impl::wconv_dot(idata, kdata, weight, odata, dot<14, tscalar>); }
                        else if (kcols == 15) { impl::wconv_dot(idata, kdata, weight, odata, dot<15, tscalar>); }
                        else if ((kcols & 0x3) == 0) { impl::wconv_dot(idata, kdata, weight, odata, dot_mod4<tscalar>); }
                        else { impl::wconv_dot(idata, kdata, weight, odata, dot_mod4x<tscalar>); }
                }
                
                // odata += idata @ kdata (using Eigen 2D blocks)
                template
                <
                        typename tmatrix,
                        typename tscalar = typename tmatrix::Scalar
                >
                void conv_eib(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
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

                // odata += weight * idata @ kdata (using Eigen 2D blocks)
                template
                <
                        typename tmatrix,
                        typename tscalar = typename tmatrix::Scalar
                >
                void wconv_eib(const tmatrix& idata, const tmatrix& kdata, tscalar weight, tmatrix& odata)
                {
                        for (auto r = 0; r < odata.rows(); r ++)
                        {
                                for (auto c = 0; c < odata.cols(); c ++)
                                {
                                        odata(r, c) += weight *
                                        kdata.cwiseProduct(idata.block(r, c, kdata.rows(), kdata.cols())).sum();
                                }
                        }
                }                
        }
}

#endif // NANOCV_CONVOLUTION_H

