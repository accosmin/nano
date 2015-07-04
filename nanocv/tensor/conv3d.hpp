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
                        const auto odims = odata.dims();

                        for (decltype(idata.dims()) i = 0; i < idata.dims(); i ++)
                        {
                                const auto imap = idata.matrix(i);
                                const auto tmap = tensor::conv2d_linearize(imap, kdata);

                                tensor::map_matrix(odata.data(), odims, osize) +=
                                tensor::map_matrix(kdata.planeData(i * odims), odims, ksize) *
                                tmap.transpose();
                        }
                }

                ///
                /// \brief gradient wrt the input
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

                        for (decltype(odata.dims()) o = 0; o < odata.dims(); o ++)
                        {
                                const auto omap = odata.matrix(o);
                                const auto tmap = tensor::corr2d_linearize(omap, kdata);

                                for (decltype(idata.dims()) i = 0; i < idata.dims(); i ++)
                                {
                                        auto imap = idata.matrix(i);
                                        const auto kmap = kdata.matrix(i * odata.dims() + o);

                                        tensor::map_vector(imap.data(), imap.size()) +=
                                        tmap *
                                        tensor::map_vector(kmap.data(), kmap.size());
                                }
                        }
                }

                ///
                /// \brief gradient wrt the parameters
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
                        const auto odims = odata.dims();

                        for (decltype(idata.dims()) i = 0; i < idata.dims(); i ++)
                        {
                                const auto imap = idata.matrix(i);
                                const auto tmap = tensor::conv2d_linearize(imap, odata.rows(), odata.cols());

                                tensor::map_matrix(kdata.planeData(i * odims), odims, ksize) =
                                tensor::map_matrix(odata.data(), odims, osize) *
                                tmap.transpose();
                        }
                }
        }
}


