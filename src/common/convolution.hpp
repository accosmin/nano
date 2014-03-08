#ifndef NANOCV_CONVOLUTION_H
#define NANOCV_CONVOLUTION_H

#include <eigen3/Eigen/Core>
#include "dot.hpp"

namespace ncv
{
        namespace math
        {
                namespace detail
		{
                        ///
                        /// 2D convolution: odata += idata @ kdata (using a column-based dot operator)
                        ///
	                template
        	        <
                                typename tdot,
	                        typename tmatrix,
	                        typename tscalar = typename tmatrix::Scalar
        	        >
                        void conv_dot(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata, tdot dot)
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
                                                        po[c] += dot(pi + c, pk, kcols);
                                                }
                                        }
                                }
                	}

                        ///
                        /// weighted 2D convolution:  += weight * idata @ kdata (using a column-based dot operator)
                        ///
	                template
        	        <
                                typename tdot,
	                        typename tmatrix,
        	                typename tscalar = typename tmatrix::Scalar
                	>
                        void wconv_dot(const tmatrix& idata, const tmatrix& kdata, tscalar weight, tmatrix& odata, tdot dot)
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
                                                        po[c] += weight * dot(pi + c, pk, kcols);
                                                }
                                        }
                                }
                	}
		}

                ///
                /// 2D convolution: odata += idata @ kdata
                ///
                template
                <
                        typename tmatrix,
                        typename tscalar = typename tmatrix::Scalar,
                        typename tindex = typename tmatrix::Index
                >
                void conv_dot(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
                {
                        detail::conv_dot(idata, kdata, odata, dot_mod4x<tscalar, tindex>);
                }

                ///
                /// 2D convolution: odata += weight * idata @ kdata
                ///
                template
                <
                        typename tmatrix,
                        typename tscalar = typename tmatrix::Scalar,
                        typename tindex = typename tmatrix::Index
                >
                void wconv_dot(const tmatrix& idata, const tmatrix& kdata, tscalar weight, tmatrix& odata)
                {
                        detail::wconv_dot(idata, kdata, weight, odata, dot_mod4x<tscalar, tindex>);
                }
                
                ///
                /// 2D convolution: odata += idata @ kdata (using Eigen 2D blocks)
                ///
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

                ///
                /// 2D convolution: odata += weight * idata @ kdata (using Eigen 2D blocks)
                ///
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

