#pragma once

#include "nanocv/tensor/matrix.hpp"

namespace ncv
{
        namespace pooling
        {
                template
                <
                        typename tscalar,
                        typename tsize
                >
                void output(
                        const tscalar* idata, tsize irows, tsize icols, tscalar alpha,
                        tscalar* wdata, tscalar* sdata, tscalar* cdata, tscalar* odata)
                {
                        const tsize orows = (irows + 1) / 2;
                        const tsize ocols = (icols + 1) / 2;
                        const tscalar ialpha = 1 / alpha;

                        auto wmap = tensor::map_matrix(wdata, irows, icols);
                        auto smap = tensor::map_matrix(sdata, orows, ocols);
                        auto cmap = tensor::map_matrix(cdata, orows, ocols);
                        auto omap = tensor::map_matrix(odata, orows, ocols);
                        auto imap = tensor::map_matrix(idata, irows, icols);

                        wmap = (imap.array() * alpha).exp();

                        smap.setZero();
                        cmap.setZero();

                        for (tsize r = 0, rr = 0; r < irows; r ++, rr = r / 2)
                        {
                                for (tsize c = 0, cc = 0; c < icols; c ++, cc = c / 2)
                                {
                                        smap(rr, cc) += wmap(r, c);
                                        cmap(rr, cc) += 1;
                                }
                        }

                        omap = ialpha * (smap.array() / cmap.array()).log();
                }

                template
                <
                        typename tscalar,
                        typename tsize
                >
                void ginput(
                        tscalar* idata, tsize irows, tsize icols,
                        const tscalar* wdata, const tscalar* sdata, const tscalar*, const tscalar* gdata)
                {
                        const tsize orows = (irows + 1) / 2;
                        const tsize ocols = (icols + 1) / 2;

                        auto wmap = tensor::map_matrix(wdata, irows, icols);
                        auto smap = tensor::map_matrix(sdata, orows, ocols);
                        auto gmap = tensor::map_matrix(gdata, orows, ocols);
                        auto imap = tensor::map_matrix(idata, irows, icols);

                        for (tsize r = 0, rr = 0; r < irows; r ++, rr = r / 2)
                        {
                                for (tsize c = 0, cc = 0; c < icols; c ++, cc = c / 2)
                                {
                                        imap(r, c) = gmap(rr, cc) * wmap(r, c) / smap(rr, cc);
                                }
                        }
                }
        }
}


