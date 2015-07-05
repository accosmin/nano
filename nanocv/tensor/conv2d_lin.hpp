#pragma once

#include "conv2d_linearize.hpp"
#include "nanocv/arch.h"

namespace ncv
{
        namespace tensor
        {
                ///
                /// \brief 2D convolution: odata += idata @ kdata (as a linear operation)
                ///     this version receives the already-linearized input matrix (idata)
                template
                <
                        typename tmatrixt
                >
                struct conv2d_lin_buf_t
                {
                        explicit conv2d_lin_buf_t(const tmatrixt& idata_linearized)
                                :       m_idata_linearized(idata_linearized)
                        {
                        }

                        template
                        <
                                typename tmatrixi,
                                typename tmatrixk = tmatrixi,
                                typename tmatrixo = tmatrixi
                        >
                        void operator()(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo&& odata) const
                        {
                                NANOCV_UNUSED1_RELEASE(idata);

                                assert(idata.rows() + 1 == kdata.rows() + odata.rows());
                                assert(idata.cols() + 1 == kdata.cols() + odata.cols());

                                tensor::map_vector(odata.data(), odata.size()) +=
                                m_idata_linearized *
                                tensor::map_vector(kdata.data(), kdata.size());
                        }

                        tmatrixt        m_idata_linearized;
                };

                ///
                /// \brief 2D convolution: odata += idata @ kdata (as a linear operation)
                ///     this version linearizes the input matrix (idata) on the fly
                ///
                struct conv2d_lin_t
                {
                        template
                        <
                                typename tmatrixi,
                                typename tmatrixk = tmatrixi,
                                typename tmatrixo = tmatrixi
                        >
                        void operator()(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata) const
                        {
                                assert(idata.rows() + 1 == kdata.rows() + odata.rows());
                                assert(idata.cols() + 1 == kdata.cols() + odata.cols());

                                const auto idata_linearized = conv2d_linearize(idata, kdata);

                                return  conv2d_lin_buf_t<decltype(idata_linearized)>(idata_linearized)
                                        (idata, kdata, odata);
                        }
                };
        }
}

