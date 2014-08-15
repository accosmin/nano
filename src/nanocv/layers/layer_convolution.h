#ifndef NANOCV_CONV_LAYER_H
#define NANOCV_CONV_LAYER_H

#include "layer.h"
#ifdef NANOCV_HAVE_OPENCL
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

                NANOCV_MAKE_CLONABLE(conv_layer_t)

                // constructor
                conv_layer_t(const string_t& parameters = string_t());

                // copy
#ifdef NANOCV_HAVE_OPENCL
                conv_layer_t(const conv_layer_t& other);
                conv_layer_t& operator=(const conv_layer_t& other);
#endif

                // resize to process new tensors of the given type
                virtual size_t resize(const tensor_t& tensor);

                // reset parameters
                virtual void zero_params();
                virtual void random_params(scalar_t min, scalar_t max);

                // serialize parameters
                virtual scalar_t* save_params(scalar_t* params) const;
                virtual const scalar_t* load_params(const scalar_t* params);

                // process inputs (compute outputs & gradients)
                virtual const tensor_t& output(const tensor_t& input);
                virtual const tensor_t& igrad(const tensor_t& output);
                virtual void pgrad(const tensor_t& output, scalar_t* gradient);

                // access functions
                virtual size_t idims() const { return m_idata.dims(); }
                virtual size_t irows() const { return m_idata.rows(); }
                virtual size_t icols() const { return m_idata.cols(); }
                virtual size_t odims() const { return m_odata.dims(); }
                virtual size_t orows() const { return m_odata.rows(); }
                virtual size_t ocols() const { return m_odata.cols(); }
                virtual size_t psize() const { return m_kdata.size(); }

        private:

                size_t krows() const { return m_kdata.rows(); }
                size_t kcols() const { return m_kdata.cols(); }

                void params_changed() const;

        private:

                // attributes
                tensor_t                m_idata;                ///< input buffer:              idims x irows x icols
                tensor_t                m_odata;                ///< output buffer:             odims x orows x ocols
                tensor_t                m_kdata;                ///< convolution kernels:       odims x idims x krows x kcols

#ifdef NANOCV_HAVE_OPENCL
                cl::Context             m_ocl_context;          ///< opencl context
                cl::CommandQueue        m_ocl_queue;            ///< opencl command queue
                cl::Program             m_ocl_program;          ///< opencl program
                cl::Kernel              m_ocl_fkernel;          ///< opencl forward kernel
                cl::Kernel              m_ocl_bikernel;         ///< opencl backward (inputs gradient) kernel
                cl::Kernel              m_ocl_bkkernel;         ///< opencl backward (convolution gradient) kernel

                cl::Buffer              m_ocl_idata;            ///< opencl buffers for various tensors
                cl::Buffer              m_ocl_kdata;

                cl::Buffer              m_ocl_gidata;
                cl::Buffer              m_ocl_gkdata;
                tensor_t                m_gkdata;

                cl::Buffer              m_ocl_odata;
#endif
        };
}

#endif // NANOCV_CONV_LAYER_H

