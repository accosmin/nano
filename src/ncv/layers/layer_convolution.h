#ifndef NANOCV_CONV_LAYER_H
#define NANOCV_CONV_LAYER_H

#include "layer.h"

namespace ncv
{
        ///
        /// \brief convolution layer
        ///
        class conv_layer_t : public layer_t
        {
        public:

                // constructor
                conv_layer_t(const string_t& params = string_t());

                NCV_MAKE_CLONABLE(conv_layer_t, layer_t,
                                  "convolution layer, parameters: dims=16[1,256],rows=8[1,32],cols=8[1,32]")

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

                /////////////////////////////////////////////////////////////////////////////////////////

        private:

                // attributes
                string_t                m_params;

                tensor_t                m_idata;        ///< input buffer:              idims x irows x icols
                tensor_t                m_odata;        ///< output buffer:             odims x orows x ocols

                tensor_t                m_kdata;        ///< convolution kernels:       odims x krows x kcols
                tensor_t                m_wdata;        ///< weights:                   1 x odims x idims

                tensor_t                m_gkdata;       ///< cumulated kernel gradients
                tensor_t                m_gwdata;       ///< cumulated weight gradients
                tensor_t                m_gidata;       ///< cumulated input gradients
        };
}

#endif // NANOCV_CONV_LAYER_H

