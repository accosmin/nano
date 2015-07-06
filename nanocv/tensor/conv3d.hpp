#pragma once

#include "conv2d_linearize.hpp"
#include "corr2d_linearize.hpp"

namespace ncv
{
        namespace tensor
        {
                ///
                /// \brief 3D convolution output: odata(o) = sum(i, idata(i) @ kdata(i, o))
                ///
                template
                <
                        typename ttensori,
                        typename ttensork,
                        typename ttensoro
                >
                void conv3d_output(const ttensori& idata, const ttensork& kdata, ttensoro&& odata)
                {
                        odata.setZero();

                        const auto osize = odata.rows() * odata.cols();
                        const auto ksize = kdata.rows() * kdata.cols();
                        const auto idims = idata.dims();
                        const auto odims = odata.dims();

                        for (decltype(idata.dims()) i = 0; i < idims; i ++)
                        {
                                const auto imap = idata.matrix(i);
                                const auto tmap = tensor::conv2d_linearize(imap, kdata);

                                tensor::map_matrix(odata.data(), odims, osize) +=
                                tensor::map_matrix(kdata.planeData(i * odims), odims, ksize) *
                                tmap;
                        }
                }

                ///
                /// \brief gradient wrt the input: odata(o) = sum(i, idata(i) @ kdata(i, o))
                ///
                template
                <
                        typename ttensori,
                        typename ttensork,
                        typename ttensoro
                >
                void conv3d_ginput(ttensori&& idata, const ttensork& kdata, const ttensoro& odata)
                {
                        idata.setZero();


                        // todo: need to transform kdata from (i, o) to (o, i) indexing

                        const auto isize = idata.rows() * idata.cols();
                        const auto ksize = kdata.rows() * kdata.cols();
                        const auto idims = idata.dims();
                        const auto odims = odata.dims();

                        for (decltype(odata.dims()) o = 0; o < odims; o ++)
                        {
                                const auto omap = odata.matrix(o);
                                const auto tmap = tensor::corr2d_linearize(omap, kdata);

                                tensor::map_matrix(idata.data(), idims, isize) +=
                                tensor::map_matrix(kdata.planeData(o * idims), idims, ksize) *
                                tmap;
                        }
                }

                ///
                /// \brief gradient wrt the parameters: odata(o) = sum(i, idata(i) @ kdata(i, o))
                ///
                template
                <
                        typename ttensori,
                        typename ttensork,
                        typename ttensoro
                >
                void conv3d_gparam(const ttensori& idata, ttensork&& kdata, const ttensoro& odata)
                {
                        const auto osize = odata.rows() * odata.cols();
                        const auto ksize = kdata.rows() * kdata.cols();
                        const auto idims = idata.dims();
                        const auto odims = odata.dims();

                        for (decltype(idata.dims()) i = 0; i < idims; i ++)
                        {
                                const auto imap = idata.matrix(i);
                                const auto tmap = tensor::conv2d_linearize(imap, odata.rows(), odata.cols());

                                tensor::map_matrix(kdata.planeData(i * odims), odims, ksize) =
                                tensor::map_matrix(odata.data(), odims, osize) *
                                tmap;
                        }
                }
        }
}


