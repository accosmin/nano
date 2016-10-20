#pragma once

#include "layer.h"

namespace nano
{
        ///
        /// \brief fully-connected convolution layer as in convolution networks.
        ///
        /// parameters:
        ///     dims    - number of output planes
        ///     rows    - convolution size
        ///     cols    - convolution size
        ///     conn    - connectivity factor: default = 1 (fully connected)
        ///     drow    - stride factor for the vertical axis: default = 1
        ///     dcol    - stride factor for the horizontal axis: default = 1
        ///
        class NANO_PUBLIC convolution_layer_t final : public layer_t
        {
        public:

                explicit convolution_layer_t(const string_t& parameters = string_t());

                virtual rlayer_t clone() const override;

                virtual tensor_size_t resize(const tensor3d_t& tensor) override;

                virtual void zero_params() override;
                virtual void random_params(scalar_t min, scalar_t max) override;

                virtual scalar_t* save_params(scalar_t* params) const override;
                virtual const scalar_t* load_params(const scalar_t* params) override;

                virtual const tensor3d_t& output(const tensor3d_t& input) override;
                virtual const tensor3d_t& ginput(const tensor3d_t& output) override;
                virtual void gparam(const tensor3d_t& output, scalar_t* gradient) override;

                virtual tensor_size_t idims() const override { return m_idata.size<0>(); }
                virtual tensor_size_t irows() const override { return m_idata.size<1>(); }
                virtual tensor_size_t icols() const override { return m_idata.size<2>(); }
                virtual tensor_size_t odims() const override { return m_odata.size<0>(); }
                virtual tensor_size_t orows() const override { return m_odata.size<1>(); }
                virtual tensor_size_t ocols() const override { return m_odata.size<2>(); }
                virtual tensor_size_t psize() const override { return m_kdata.size() + m_bdata.size(); }
                virtual tensor_size_t flops() const override
                {
                        return (idims() * odims() / kconn()) * (orows() * ocols()) * (krows() * kcols());
                }

                tensor_size_t kconn() const { return m_kconn; }
                tensor_size_t krows() const { return m_kdata.size<2>(); }
                tensor_size_t kcols() const { return m_kdata.size<3>(); }

                tensor_size_t drows() const { return m_drows; }
                tensor_size_t dcols() const { return m_dcols; }

                const tensor4d_t& kdata() const { return m_kdata; }
                const vector_t& bdata() const { return m_bdata; }

        private:

                void params_changed();

        private:

                // attributes
                tensor3d_t      m_idata;        ///< input buffer:              idims x irows x icols
                tensor3d_t      m_odata;        ///< output buffer:             odims x orows x ocols
                tensor_size_t   m_kconn;        ///< input connectivity factor
                tensor_size_t   m_drows;        ///< stride factor
                tensor_size_t   m_dcols;        ///< stride factor
                tensor4d_t      m_kdata;        ///< convolution kernels:       odims x (idims/kconn) x krows x kcols
                vector_t        m_bdata;        ///< convolution bias:          odims

                tensor3d_t      m_idata_toe;    ///< toeplitz-like matrices:    idims x (krows*kcols) x (orows*ocols)
                tensor4d_t      m_kdata_inv;    ///< convolution kernels:       idims x (odims/kconn) x krows x kcols

                // todo: these should be removed! use directly Eigen calls to map the output buffers!
                matrix_t        m_toe_oodata;
                matrix_t        m_toe_iidata, m_toe_iodata;
                matrix_t        m_toe_kkdata, m_toe_kodata;
        };
}
