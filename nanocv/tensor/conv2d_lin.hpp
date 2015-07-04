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
                ///
                template
                <
                        typename tmatrixi,
                        typename tmatrixk = tmatrixi,
                        typename tmatrixt = tmatrixi,
                        typename tmatrixo = tmatrixi
                >
                void conv2d_lin_buffered(const tmatrixi& idata,
                        const tmatrixk& kdata, tmatrixt&& idata_linearized, tmatrixo&& odata)
                {
                        NANOCV_UNUSED1_RELEASE(idata);

                        assert(idata.rows() + 1 == kdata.rows() + odata.rows());
                        assert(idata.cols() + 1 == kdata.cols() + odata.cols());

                        tensor::map_vector(odata.data(), odata.size()) +=
                        idata_linearized *
                        tensor::map_vector(kdata.data(), kdata.size());
                }

                ///
                /// \brief 2D convolution: odata += idata @ kdata (as a linear operation)
                ///     this version linearizes the input matrix (idata) on the fly
                ///
                template
                <
                        typename tmatrixi,
                        typename tmatrixk = tmatrixi,
                        typename tmatrixo = tmatrixi
                >
                void conv2d_lin(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata)
                {
                        assert(idata.rows() + 1 == kdata.rows() + odata.rows());
                        assert(idata.cols() + 1 == kdata.cols() + odata.cols());

                        return conv2d_lin_buffered(idata, kdata, conv2d_linearize(idata, kdata), odata);
                }
        }
}

