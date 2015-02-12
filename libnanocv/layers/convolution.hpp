#pragma once

#include "libnanocv/util/conv2d.hpp"
#include "libnanocv/util/corr2d.hpp"
#include "libnanocv/tensor/matrix.hpp"

namespace ncv
{
        namespace convolution
        {
                template
                <
                        typename tscalar,
                        typename tsize
                >
                void output(
                        const tscalar* idata, tsize idims,
                        const tscalar* kdata, tsize krows, tsize kcols,
                        tscalar* odata, tsize odims, tsize orows, tsize ocols)
                {
                        const tsize irows = orows + krows - 1;
                        const tsize icols = ocols + kcols - 1;
                        const tsize isize = irows * icols;

                        const tsize osize = orows * ocols;
                        const tsize ksize = krows * kcols;

                        // output
                        for (tsize o = 0, k = 0; o < odims; o ++)
                        {
                                auto omap = tensor::make_matrix(odata + o * osize, orows, ocols);

                                omap.setZero();
                                for (tsize i = 0; i < idims; i ++, k ++)
                                {
                                        auto imap = tensor::make_matrix(idata + i * isize, irows, icols);
                                        auto kmap = tensor::make_matrix(kdata + k * ksize, krows, kcols);

                                        ncv::conv2d_dyn(imap, kmap, omap);
                                }
                        }
                }

                template
                <
                        typename tscalar,
                        typename tsize
                >
                void ginput(
                        tscalar* gidata, tsize idims,
                        const tscalar* kdata, tsize krows, tsize kcols,
                        const tscalar* odata, tsize odims, tsize orows, tsize ocols)
                {
                        const tsize irows = orows + krows - 1;
                        const tsize icols = ocols + kcols - 1;
                        const tsize isize = irows * icols;

                        const tsize osize = orows * ocols;
                        const tsize ksize = krows * kcols;

                        // input gradient
                        tensor::make_vector(gidata, idims * isize).setZero();
                        for (tsize o = 0, k = 0; o < odims; o ++)
                        {
                                auto omap = tensor::make_matrix(odata + o * osize, orows, ocols);

                                for (tsize i = 0; i < idims; i ++, k ++)
                                {
                                        auto gimap = tensor::make_matrix(gidata + i * isize, irows, icols);
                                        auto kmap = tensor::make_matrix(kdata + k * ksize, krows, kcols);

                                        ncv::corr2d_dyn(omap, kmap, gimap);
                                }
                        }
                }

                template
                <
                        typename tscalar,
                        typename tsize
                >
                void gparam(
                        const tscalar* idata, tsize idims,
                        tscalar* gkdata, tsize krows, tsize kcols,
                        const tscalar* odata, tsize odims, tsize orows, tsize ocols)
                {
                        const tsize irows = orows + krows - 1;
                        const tsize icols = ocols + kcols - 1;
                        const tsize isize = irows * icols;

                        const tsize osize = orows * ocols;
                        const tsize ksize = krows * kcols;

                        // convolution kernel gradient
                        for (tsize o = 0, k = 0; o < odims; o ++)
                        {
                                auto omap = tensor::make_matrix(odata + o * osize, orows, ocols);

                                for (tsize i = 0; i < idims; i ++, k ++)
                                {
                                        auto imap = tensor::make_matrix(idata + i * isize, irows, icols);
                                        auto gkmap = tensor::make_matrix(gkdata + k * ksize, krows, kcols);

                                        gkmap.setZero();
                                        ncv::conv2d_dyn(imap, omap, gkmap);
                                }
                        }
                }
        }
}


