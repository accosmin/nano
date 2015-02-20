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
                        tensor::map_vector(odata, osize).noalias() =
                                tensor::map_vector(bdata, osize) +
                                tensor::map_matrix(wdata, osize, isize) *
                                tensor::map_vector(idata, isize);
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
                        tensor::map_vector(idata, isize).noalias() =
                                tensor::map_matrix(wdata, osize, isize).transpose() *
                                tensor::map_vector(odata, osize);
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
                        tensor::map_vector(gbdata, osize).noalias() =
                                tensor::map_vector(odata, osize);

                        tensor::map_matrix(gwdata, osize, isize).noalias() =
                                tensor::map_vector(odata, osize) *
                                tensor::map_vector(idata, isize).transpose();
                }
        }
}

