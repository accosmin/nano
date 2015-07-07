#pragma once

#include "conv2d.hpp"
#include "corr2d.hpp"

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
                        typename ttensoro,
                        typename tsize = typename ttensori::Index
                >
                void conv3d_output(const ttensori& idata, const ttensork& kdata, ttensoro&& odata)
                {
                        const auto osize = odata.planeSize();
                        const auto ksize = kdata.planeSize();
                        const auto idims = idata.dims();
                        const auto odims = odata.dims();

                        conv2d_linearizer_t<typename ttensori::Scalar> conv2dlin;

                        odata.setZero();
                        for (tsize i = 0; i < idims; i ++)
                        {
                                const auto imap = idata.matrix(i);

                                tensor::map_matrix(odata.data(), odims, osize) +=
                                tensor::map_matrix(kdata.planeData(i * odims), odims, ksize) *
                                conv2dlin(imap, kdata);
                        }
                }

                ///
                /// \brief gradient wrt the input: odata(o) = sum(i, idata(i) @ kdata(i, o))
                ///
                template
                <
                        typename ttensori,
                        typename ttensork,
                        typename ttensoro,
                        typename tsize = typename ttensork::Index
                >
                void conv3d_ginput(ttensori&& idata, const ttensork& kdata, const ttensoro& odata)
                {
                        idata.setZero();

                        // todo: need to transform kdata from (i, o) to (o, i) indexing

                        const auto isize = idata.rows() * idata.cols();
                        const auto ksize = kdata.rows() * kdata.cols();
                        const auto idims = idata.dims();
                        const auto odims = odata.dims();

                        corr2d_linearizer_t<typename ttensork::Scalar> corr2lin;

                        for (decltype(odata.dims()) o = 0; o < odims; o ++)
                        {
                                const auto omap = odata.matrix(o);

                                tensor::map_matrix(idata.data(), idims, isize) +=
                                tensor::map_matrix(kdata.planeData(o * idims), idims, ksize) *
                                corr2lin(omap, kdata);
                        }
                }

                ///
                /// \brief gradient wrt the parameters: odata(o) = sum(i, idata(i) @ kdata(i, o))
                ///
                template
                <
                        typename ttensori,
                        typename ttensork,
                        typename ttensoro,
                        typename tsize = typename ttensoro::Index
                >
                void conv3d_gparam(const ttensori& idata, ttensork&& kdata, const ttensoro& odata)
                {
                        const auto osize = odata.rows() * odata.cols();
                        const auto ksize = kdata.rows() * kdata.cols();
                        const auto idims = idata.dims();
                        const auto odims = odata.dims();

                        conv2d_linearizer_t<typename ttensoro::Scalar> conv2dlin;

                        for (decltype(idata.dims()) i = 0; i < idims; i ++)
                        {
                                const auto imap = idata.matrix(i);

                                tensor::map_matrix(kdata.planeData(i * odims), odims, ksize) =
                                tensor::map_matrix(odata.data(), odims, osize) *
                                conv2dlin(imap, odata);
                        }
                }
        }
}


