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
                        typename ttensor
                >
                class conv3d_t
                {
                public:

                        typedef typename ttensor::Index                                 tsize;
                        typedef typename ttensor::Scalar                                tscalar;
                        typedef typename tensor::matrix_types_t<tscalar>::tmatrix       tmatrix;

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
                        bool output(const ttensori& idata, ttensoro&& odata);

                        ///
                        /// \brief gradient wrt the input
                        ///
                        template
                        <
                                typename ttensori,
                                typename ttensoro
                        >
                        bool ginput(ttensori&& idata, const ttensoro& odata);

                        ///
                        /// \brief gradient wrt the parameters
                        ///
                        template
                        <
                                typename ttensori,
                                typename ttensork,      ///< assumming of (i, o) type
                                typename ttensoro
                        >
                        bool gparam(const ttensori& idata, ttensork&& kdata, const ttensoro& odata);

                private:

                        template
                        <
                                typename ttensori,
                                typename ttensoro
                        >
                        bool check_inputs(ttensori&& idata, ttensoro&& odata) const
                        {
                                return  idata.dims() == m_idims &&
                                        odata.dims() == m_odims;
                        }

                private:

                        // attributes
                        tsize           m_idims;        ///< number of input dimensions/planes
                        tsize           m_odims;        ///< number of output dimensions/planes

                        ttensor         m_kdata_io;     ///< convolutions organized as (input i, output o)
                        ttensor         m_kdata_oi;     ///< convolutions organized as (output o, input i)

                        tmatrix         m_idata_lin;    ///< buffer to store linearized inputs
                        tmatrix         m_odata_lin;    ///< buffer to store linearized outputs
                        tmatrix         m_kdata_lin;    ///< buffer to store linearized convolutions
                };

                template
                <
                        typename ttensor
                >
                bool conv3d_t<ttensor>::reset(const ttensor& kdata_io, const tsize idims, const tsize odims)
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
                        typename ttensor
                >
                template
                <
                        typename ttensori,
                        typename ttensoro
                >
                bool conv3d_t<ttensor>::output(const ttensori& idata, ttensoro&& odata)
                {
                        if (!check_inputs(idata, odata))
                        {
                                return false;
                        }

                        const auto& kdata = m_kdata_oi;

                        const auto osize = odata.planeSize();
                        const auto ksize = kdata.planeSize();

                        m_idata_lin.resize(m_idims * ksize, osize);
                        for (tsize i = 0; i < m_idims; i ++)
                        {
                                linearize_conv2d(
                                        idata.matrix(i), kdata.rows(), kdata.cols(),
                                        m_idata_lin.block(i * ksize, 0, ksize, osize));
                        }

                        tensor::map_matrix(odata.data(), m_odims, osize) =
                        tensor::map_matrix(kdata.data(), m_odims, m_idims * ksize) *
                        m_idata_lin;

                        return true;
                }

                template
                <
                        typename ttensor
                >
                template
                <
                        typename ttensori,
                        typename ttensoro
                >
                bool conv3d_t<ttensor>::ginput(ttensori&& idata, const ttensoro& odata)
                {
                        if (!check_inputs(idata, odata))
                        {
                                return false;
                        }

                        const auto& kdata = m_kdata_io;

                        const auto isize = idata.planeSize();
                        const auto ksize = kdata.planeSize();

                        m_odata_lin.resize(m_odims * ksize, isize);
                        for (tsize o = 0; o < m_odims; o ++)
                        {
                                linearize_corr2d(
                                        odata.matrix(o), kdata.rows(), kdata.cols(),
                                        m_odata_lin.block(o * ksize, 0, ksize, isize));
                        }

                        tensor::map_matrix(idata.data(), m_idims, isize) =
                        tensor::map_matrix(kdata.data(), m_idims, m_odims * ksize) *
                        m_odata_lin;

                        return true;
                }

                template
                <
                        typename ttensor
                >
                template
                <
                        typename ttensori,
                        typename ttensork,
                        typename ttensoro
                >
                bool conv3d_t<ttensor>::gparam(const ttensori& idata, ttensork&& kdata, const ttensoro& odata)
                {
                        if (    !check_inputs(idata, odata) ||
                                kdata.dims() != m_idims * m_odims)
                        {
                                return false;
                        }

                        const auto osize = odata.planeSize();
                        const auto ksize = kdata.planeSize();

                        m_kdata_lin.resize(osize, ksize);
                        for (tsize i = 0; i < m_idims; i ++)
                        {
                                linearize_conv2d(
                                        idata.matrix(i), odata.rows(), odata.cols(),
                                        m_kdata_lin);

                                tensor::map_matrix(kdata.planeData(i * m_odims), m_odims, ksize) =
                                tensor::map_matrix(odata.data(), m_odims, osize) *
                                m_kdata_lin;
                        }

                        return true;
                }
        }
}


