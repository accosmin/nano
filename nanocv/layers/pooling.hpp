#pragma once

#include "nanocv/tensor/matrix.hpp"

namespace ncv
{
        namespace pooling
        {
                template
                <
                        typename tmatrixi,
                        typename tscalar,
                        typename tmatrixw,
                        typename tmatrixs,
                        typename tmatrixc,
                        typename tmatrixo
                >
                void output(
                        const tmatrixi& idata, tscalar alpha,
                        tmatrixw&& wdata, tmatrixs&& sdata, tmatrixc&& cdata, tmatrixo&& odata)
                {
                        const auto irows = idata.rows();
                        const auto icols = idata.cols();
                        const tscalar ialpha = 1 / alpha;

                        wdata = (idata.array() * alpha).exp();

                        sdata.setZero();
                        cdata.setZero();

                        for (auto r = 0, rr = 0; r < irows; r ++, rr = r / 2)
                        {
                                for (auto c = 0, cc = 0; c < icols; c ++, cc = c / 2)
                                {
                                        sdata(rr, cc) += wdata(r, c);
                                        cdata(rr, cc) += 1;
                                }
                        }

                        odata = ialpha * (sdata.array() / cdata.array()).log();
                }

                template
                <
                        typename tmatrixi,
                        typename tmatrixw,
                        typename tmatrixs,
                        typename tmatrixc,
                        typename tmatrixo
                >
                void ginput(
                        tmatrixi&& gidata,
                        const tmatrixw& wdata, const tmatrixs& sdata, const tmatrixc&, const tmatrixo& odata)
                {
                        const auto irows = gidata.rows();
                        const auto icols = gidata.cols();

                        for (auto r = 0, rr = 0; r < irows; r ++, rr = r / 2)
                        {
                                for (auto c = 0, cc = 0; c < icols; c ++, cc = c / 2)
                                {
                                        gidata(r, c) = odata(rr, cc) * wdata(r, c) / sdata(rr, cc);
                                }
                        }
                }
        }
}


