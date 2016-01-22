#pragma once

#include <cassert>

namespace tensor
{
        ///
        /// \brief check matrices for the convolution: odata += idata @ kdata
        ///
        template
        <
                typename tmatrixi,
                typename tmatrixk,
                typename tmatrixo
        >
        void conv2d_assert(const tmatrixi& idata, const tmatrixk& kdata, const tmatrixo& odata)
        {
                assert(idata.rows() + 1 == kdata.rows() + odata.rows());
                assert(idata.cols() + 1 == kdata.cols() + odata.cols());

                (void)(idata);
                (void)(kdata);
                (void)(odata);
        }
}

