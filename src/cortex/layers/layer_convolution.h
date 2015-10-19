#pragma once

#include "cortex/layer.h"
#include "tensor/conv3d_lin.hpp"

namespace cortex
{
        ///
        /// \brief convolution layer
        ///
        /// parameters:
        ///     dims=16[1,256]          - number of convolutions (output dimension)
        ///     rows=8[1,32]            - convolution size
        ///     cols=8[1,32]            - convolution size
        ///
        class conv_layer_t : public layer_t
        {
        public:

                NANOCV_MAKE_CLONABLE(conv_layer_t,
                                     "convolution layer, "\
                                     "parameters: dims=16[1,256],rows=8[1,32],cols=8[1,32]")

                // constructor
                explicit conv_layer_t(const string_t& parameters = string_t());

                // destructor
                virtual ~conv_layer_t();

                // resize to process new tensors of the given type
                virtual tensor_size_t resize(const tensor_t& tensor) override;

                // reset parameters
                virtual void zero_params() override;
                virtual void random_params(scalar_t min, scalar_t max) override;

                // serialize parameters
                virtual scalar_t* save_params(scalar_t* params) const override;
                virtual const scalar_t* load_params(const scalar_t* params) override;

                // process inputs (compute outputs & gradients)
                virtual const tensor_t& output(const tensor_t& input) override;
                virtual const tensor_t& ginput(const tensor_t& output) override;
                virtual void gparam(const tensor_t& output, scalar_t* gradient) override;

                // access functions
                virtual tensor_size_t idims() const override { return m_idata.dims(); }
                virtual tensor_size_t irows() const override { return m_idata.rows(); }
                virtual tensor_size_t icols() const override { return m_idata.cols(); }
                virtual tensor_size_t odims() const override { return m_odata.dims(); }
                virtual tensor_size_t orows() const override { return m_odata.rows(); }
                virtual tensor_size_t ocols() const override { return m_odata.cols(); }
                virtual tensor_size_t psize() const override;

                // flops
                virtual tensor_size_t output_flops() const override { return odims() * idims() * oppsize() * kppsize(); }
                virtual tensor_size_t ginput_flops() const override { return odims() * orows() * oppsize() * kppsize(); }
                virtual tensor_size_t gparam_flops() const override { return odims() * orows() * ippsize() * oppsize(); }

        private:

                tensor_size_t kdims() const { return m_kdata.dims(); }
                tensor_size_t krows() const { return m_kdata.rows(); }
                tensor_size_t kcols() const { return m_kdata.cols(); }
                tensor_size_t ksize() const { return m_kdata.size(); }

                tensor_size_t oppsize() const { return m_odata.planeSize(); }
                tensor_size_t ippsize() const { return m_idata.planeSize(); }
                tensor_size_t kppsize() const { return m_kdata.planeSize(); }

                void params_changed();

        private:

                // attributes
                tensor_t                m_idata;        ///< input buffer:              idims x irows x icols
                tensor_t                m_odata;        ///< output buffer:             odims x orows x ocols
                tensor_t                m_kdata;        ///< convolution kernels:       idims x odims x krows x kcols
                tensor_t                m_bdata;        ///< convolution bias:          odims x 1 x 1

                tensor::conv3d_lin_t
                <tensor_t>              m_kconv;        ///< convolution kernel
        };
}
