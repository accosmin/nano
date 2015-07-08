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
                        typename ttensor,
                        typename tsize = typename ttensor::Index,
                        typename tscalar = typename ttensor::Scalar
                >
                class conv3d_t
                {
                public:

                        ///
                        /// \brief constructor
                        ///
                        conv3d_t()
                                :       m_idims(0),
                                        m_odims(0)
                        {
                        }

                        ///
                        /// \brief change convolutions
                        ///
                        bool reset(const ttensor& kdata_io, const tsize idims, const tsize odims);

                        ///
                        /// \brief output
                        ///
                        template
                        <
                                typename ttensori,
                                typename ttensoro
                        >
                        bool output(const ttensori& idata, ttensoro&& odata) const;

                        ///
                        /// \brief gradient wrt the input
                        ///
                        template
                        <
                                typename ttensori,
                                typename ttensoro
                        >
                        bool ginput(ttensori&& idata, const ttensoro& odata) const;

                        ///
                        /// \brief gradient wrt the parameters
                        ///
                        template
                        <
                                typename ttensori,
                                typename ttensork,      ///< assumming of (i, o) type
                                typename ttensoro
                        >
                        bool gparam(const ttensori& idata, ttensork&& kdata, const ttensoro& odata) const;

                        const ttensor& kdata() const
                        {
                                return m_kdata_io;
                        }

                private:

                        // attributes
                        ttensor         m_kdata_io;     ///< convolutions organized as (input i, output o)
                        ttensor         m_kdata_oi;     ///< convolutions organized as (output o, input i)
                        tsize           m_idims;
                        tsize           m_odims;
                };

                template
                <
                        typename ttensor,
                        typename tsize,
                        typename tscalar
                >
                bool conv3d_t<ttensor, tsize, tscalar>::reset(
                        const ttensor& kdata_io, const tsize idims, const tsize odims)
                {
                        if (idims * odims != kdata_io.dims())
                        {
                                return false;
                        }

                        else
                        {
                                m_idims = idims;
                                m_odims = odims;

                                m_kdata_io = kdata_io;
                                m_kdata_oi = kdata_io;

                                for (tsize o = 0; o < m_odims; o ++)
                                {
                                        for (tsize i = 0; i < m_idims; i ++)
                                        {
                                                m_kdata_oi.vector(o * m_idims + i) =
                                                m_kdata_io.vector(i * m_odims + o);
                                        }
                                }

                                return true;
                        }
                }

                template
                <
                        typename ttensor,
                        typename tsize,
                        typename tscalar
                >
                template
                <
                        typename ttensori,
                        typename ttensoro
                >
                bool conv3d_t<ttensor, tsize, tscalar>::output(const ttensori& idata, ttensoro&& odata) const
                {
                        if (    idata.dims() != m_idims ||
                                odata.dims() != m_odims)
                        {
                                return false;
                        }

                        const auto& kdata = m_kdata_oi;

                        const auto osize = odata.planeSize();
                        const auto ksize = kdata.planeSize();

                        typedef typename tensor::matrix_types_t<tscalar>::tmatrix tmatrix;

                        tmatrix idata_lin(m_idims * ksize, osize);

                        conv2d_linearizer_t<tscalar> conv2dlin;
                        for (tsize i = 0; i < m_idims; i ++)
                        {
                                idata_lin.block(i * ksize, 0, ksize, osize) =
                                conv2dlin(idata.matrix(i), kdata);
                        }

                        tensor::map_matrix(odata.data(), m_odims, osize) =
                        tensor::map_matrix(kdata.data(), m_odims, m_idims * ksize) *
                        idata_lin;

                        return true;
                }

                template
                <
                        typename ttensor,
                        typename tsize,
                        typename tscalar
                >
                template
                <
                        typename ttensori,
                        typename ttensoro
                >
                bool conv3d_t<ttensor, tsize, tscalar>::ginput(
                        ttensori&& idata, const ttensoro& odata) const
                {
                        if (    idata.dims() != m_idims ||
                                odata.dims() != m_odims)
                        {
                                return false;
                        }

                        const auto& kdata = m_kdata_io;

                        const auto isize = idata.planeSize();
                        const auto ksize = kdata.planeSize();

                        typedef typename tensor::matrix_types_t<tscalar>::tmatrix tmatrix;

                        tmatrix odata_lin(m_odims * ksize, isize);

                        corr2d_linearizer_t<tscalar> corr2lin;
                        for (tsize o = 0; o < m_odims; o ++)
                        {
                                odata_lin.block(o * ksize, 0, ksize, isize) =
                                corr2lin(odata.matrix(o), kdata);
                        }

                        tensor::map_matrix(idata.data(), m_idims, isize) =
                        tensor::map_matrix(kdata.data(), m_idims, m_odims * ksize) *
                        odata_lin;

                        return true;
                }

                template
                <
                        typename ttensor,
                        typename tsize,
                        typename tscalar
                >
                template
                <
                        typename ttensori,
                        typename ttensork,
                        typename ttensoro
                >
                bool conv3d_t<ttensor, tsize, tscalar>::gparam(
                        const ttensori& idata, ttensork&& kdata, const ttensoro& odata) const
                {
                        if (    idata.dims() != m_idims ||
                                odata.dims() != m_odims ||
                                kdata.dims() != m_idims * m_odims)
                        {
                                return false;
                        }

                        const auto osize = odata.planeSize();
                        const auto ksize = kdata.planeSize();

                        conv2d_linearizer_t<tscalar> conv2dlin;

                        for (tsize i = 0; i < m_idims; i ++)
                        {
                                tensor::map_matrix(kdata.planeData(i * m_odims), m_odims, ksize) =
                                tensor::map_matrix(odata.data(), m_odims, osize) *
                                conv2dlin(idata.matrix(i), odata);
                        }

                        return true;
                }
        }
}


