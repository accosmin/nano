#pragma once

#include "common/conv2d.hpp"
#include "common/corr2d.hpp"
#include "tensor/matrix.hpp"

namespace ncv
{
        namespace convolution
        {
                template
                <
                        typename tscalar
                >
                bool is_masked(tscalar value)
                {
                        return value > static_cast<tscalar>(0.5);
                }

                template
                <
                        typename tscalar,
                        typename tsize
                >
                void output(
                        const tscalar* mdata,
                        const tscalar* idata, tsize idims,
                        const tscalar* kdata, tsize krows, tsize kcols,
                        tscalar* odata, tsize odims, tsize orows, tsize ocols)
                {
                        const tsize irows = orows + krows - 1;
                        const tsize icols = ocols + kcols - 1;
                        const tsize isize = irows * icols;

                        const tsize osize = orows * ocols;
                        const tsize ksize = krows * kcols;

                        auto mmap = tensor::make_matrix(mdata, odims, idims);

                        // output
                        for (tsize o = 0, k = 0; o < odims; o ++)
                        {
                                auto omap = tensor::make_matrix(odata + o * osize, orows, ocols);

                                omap.setZero();
                                for (tsize i = 0; i < idims; i ++, k ++)
                                {
                                        auto imap = tensor::make_matrix(idata + i * isize, irows, icols);
                                        auto kmap = tensor::make_matrix(kdata + k * ksize, krows, kcols);

                                        if (is_masked(mmap(o, i)))
                                        {
                                                ncv::conv2d_dyn(imap, kmap, omap);
                                        }
                                }
                        }
                }

                template
                <
                        typename tscalar,
                        typename tsize
                >
                void ginput(
                        const tscalar* mdata,
                        tscalar* gidata, tsize idims,
                        const tscalar* kdata, tsize krows, tsize kcols,
                        const tscalar* odata, tsize odims, tsize orows, tsize ocols)
                {
                        const tsize irows = orows + krows - 1;
                        const tsize icols = ocols + kcols - 1;
                        const tsize isize = irows * icols;

                        const tsize osize = orows * ocols;
                        const tsize ksize = krows * kcols;

                        auto mmap = tensor::make_matrix(mdata, odims, idims);

                        // input gradient
                        tensor::make_vector(gidata, idims * isize).setZero();
                        for (tsize o = 0, k = 0; o < odims; o ++)
                        {
                                auto omap = tensor::make_matrix(odata + o * osize, orows, ocols);

                                for (tsize i = 0; i < idims; i ++, k ++)
                                {
                                        auto gimap = tensor::make_matrix(gidata + i * isize, irows, icols);
                                        auto kmap = tensor::make_matrix(kdata + k * ksize, krows, kcols);

                                        if (is_masked(mmap(o, i)))
                                        {
                                                ncv::corr2d_dyn(omap, kmap, gimap);
                                        }
                                }
                        }
                }

                template
                <
                        typename tscalar,
                        typename tsize
                >
                void gparam(
                        const tscalar* mdata,
                        const tscalar* idata, tsize idims,
                        tscalar* gkdata, tsize krows, tsize kcols,
                        const tscalar* odata, tsize odims, tsize orows, tsize ocols)
                {
                        const tsize irows = orows + krows - 1;
                        const tsize icols = ocols + kcols - 1;
                        const tsize isize = irows * icols;

                        const tsize osize = orows * ocols;
                        const tsize ksize = krows * kcols;

                        auto mmap = tensor::make_matrix(mdata, odims, idims);

                        // convolution kernel gradient
                        for (tsize o = 0, k = 0; o < odims; o ++)
                        {
                                auto omap = tensor::make_matrix(odata + o * osize, orows, ocols);

                                for (tsize i = 0; i < idims; i ++)
                                {
                                        auto imap = tensor::make_matrix(idata + i * isize, irows, icols);
                                        auto gkmap = tensor::make_matrix(gkdata + k * ksize, krows, kcols);

                                        if (is_masked(mmap(o, i)))
                                        {
                                                gkmap.setZero();
                                                ncv::conv2d_dyn(imap, omap, gkmap);
                                                k ++;
                                        }
                                }
                        }
                }
        }
}


