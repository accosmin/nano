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
        class convolution_layer_t : public layer_t
        {
        public:

                NANO_MAKE_CLONABLE(convolution_layer_t,
                        "convolution layer (implemented using Toeplitz matrices)",
                        "dims=16[1,256],rows=8[1,32],cols=8[1,32],conn=1[1,16],drow=1[1,8],dcol=1[1,8]")

                // constructor
                explicit convolution_layer_t(const string_t& parameters = string_t());

                // resize to process new tensors of the given type
                virtual tensor_size_t resize(const tensor3d_t& tensor) override;

                // reset parameters
                virtual void zero_params() override;
                virtual void random_params(scalar_t min, scalar_t max) override;

                // serialize parameters
                virtual scalar_t* save_params(scalar_t* params) const override;
                virtual const scalar_t* load_params(const scalar_t* params) override;

                // process inputs (compute outputs & gradients)
                virtual const tensor3d_t& output(const tensor3d_t& input) override;
                virtual const tensor3d_t& ginput(const tensor3d_t& output) override;
                virtual void gparam(const tensor3d_t& output, scalar_t* gradient) override;

                // access functions
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

        private:

                tensor_size_t kconn() const { return m_kconn; }
                tensor_size_t krows() const { return m_kdata.size<2>(); }
                tensor_size_t kcols() const { return m_kdata.size<3>(); }

        private:

                // attributes
                tensor3d_t      m_idata;        ///< input buffer:              idims x irows x icols
                tensor3d_t      m_odata;        ///< output buffer:             odims x orows x ocols
                tensor_size_t   m_kconn;        ///< input connectivity factor
                tensor_size_t   m_drows;        ///< stride factor
                tensor_size_t   m_dcols;        ///< stride factor
                tensor4d_t      m_kdata;        ///< convolution kernels:       odims x (idims/kconn) x krows x kcols
                vector_t        m_bdata;        ///< convolution bias:          odims

                matrix_t        m_toe_oidata, m_toe_oodata, m_toe_okdata;
                matrix_t        m_toe_iidata, m_toe_iodata, m_toe_ikdata;
                matrix_t        m_toe_kidata, m_toe_kkdata, m_toe_kodata;
        };
}
