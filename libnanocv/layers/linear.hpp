#pragma once

#include "libnanocv/tensor/vector.hpp"
#include "libnanocv/tensor/matrix.hpp"

namespace ncv
{
        namespace linear
        {
                template
                <
                        typename tscalar,
                        typename tsize
                >
                void output(
                        const tscalar* idata, tsize isize,
                        const tscalar* wdata,
                        const tscalar* bdata,
                        tscalar* odata, tsize osize)
                {
                        // output
                        tensor::make_vector(odata, osize).noalias() =
                                tensor::make_vector(bdata, osize) +
                                tensor::make_matrix(wdata, osize, isize) *
                                tensor::make_vector(idata, isize);
                }

                template
                <
                        typename tscalar,
                        typename tsize
                >
                void ginput(
                        tscalar* idata, tsize isize,
                        const tscalar* wdata,
                        const tscalar* odata, tsize osize)
                {
                        // input gradient
                        tensor::make_vector(idata, isize).noalias() =
                                tensor::make_matrix(wdata, osize, isize).transpose() *
                                tensor::make_vector(odata, osize);
                }

                template
                <
                        typename tscalar,
                        typename tsize
                >
                void gparam(
                        tscalar* idata, tsize isize,
                        tscalar* gwdata,
                        tscalar* gbdata,
                        const tscalar* odata, tsize osize)
                {
                        // bias & weights gradient
                        tensor::make_vector(gbdata, osize).noalias() =
                                tensor::make_vector(odata, osize);

                        tensor::make_matrix(gwdata, osize, isize).noalias() =
                                tensor::make_vector(odata, osize) *
                                tensor::make_vector(idata, isize).transpose();
                }
        }
}

