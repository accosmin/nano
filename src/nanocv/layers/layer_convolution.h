#ifndef NANOCV_CONV_LAYER_H
#define NANOCV_CONV_LAYER_H

#include "layer.h"
#if 0
#include "opencl/opencl.h"
#endif

namespace ncv
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

                // constructor
                conv_layer_t(const string_t& parameters = string_t());

                // create an object clone
                virtual rlayer_t clone(const string_t& parameters) const
                {
                        return rlayer_t(new conv_layer_t(parameters));
                }

                // resize to process new tensors of the given type
                virtual size_t resize(const tensor_t& tensor);

                // reset parameters
                virtual void zero_params();
                virtual void random_params(scalar_t min, scalar_t max);

                // serialize parameters & gradients
                virtual ovectorizer_t& save_params(ovectorizer_t& s) const;
                virtual ovectorizer_t& save_grad(ovectorizer_t& s) const;
                virtual ivectorizer_t& load_params(ivectorizer_t& s);

                // process inputs (compute outputs & gradients)
                virtual const tensor_t& forward(const tensor_t& input);
                virtual const tensor_t& backward(const tensor_t& gradient);

                // access functions
                virtual const tensor_t& input() const { return m_idata; }
                virtual const tensor_t& output() const { return m_odata; }

        private:

                /////////////////////////////////////////////////////////////////////////////////////////

                size_t idims() const { return m_idata.dims(); }
                size_t irows() const { return m_idata.rows(); }
                size_t icols() const { return m_idata.cols(); }

                size_t odims() const { return m_odata.dims(); }
                size_t orows() const { return m_odata.rows(); }
                size_t ocols() const { return m_odata.cols(); }

                size_t krows() const { return m_kdata.rows(); }
                size_t kcols() const { return m_kdata.cols(); }

                void params_changed() const;

                /////////////////////////////////////////////////////////////////////////////////////////

        private:

                // attributes
                tensor_t                m_idata;                ///< input buffer:              idims x irows x icols
                tensor_t                m_odata;                ///< output buffer:             odims x orows x ocols
                tensor_t                m_kdata;                ///< convolution kernels:       odims x krows x kcols
                tensor_t                m_wdata;                ///< weights:                   1 x odims x idims

                tensor_t                m_gkdata;               ///< cumulated kernel gradients
                tensor_t                m_gwdata;               ///< cumulated weight gradients
                tensor_t                m_gidata;               ///< cumulated input gradients

#if 0
                cl::CommandQueue        m_ocl_queue;            ///< opencl command queue
                cl::Program             m_ocl_program;          ///< opencl program
                cl::Kernel              m_ocl_fkernel;          ///< opencl forward kernel
                cl::Kernel              m_ocl_bikernel;         ///< opencl backward (inputs gradient) kernel
                cl::Kernel              m_ocl_bkkernel;         ///< opencl backward (convolution gradient) kernel
                cl::Kernel              m_ocl_bwkernel;         ///< opencl backward (weights gradient) kernel

                cl::Buffer              m_ocl_idata;            ///< opencl buffers for various tensors
                cl::Buffer              m_ocl_kdata;
                cl::Buffer              m_ocl_wdata;

                cl::Buffer              m_ocl_gidata;
                cl::Buffer              m_ocl_gkdata;
                cl::Buffer              m_ocl_gwdata;

                cl::Buffer              m_ocl_odata;
#endif
        };
}

#endif // NANOCV_CONV_LAYER_H

