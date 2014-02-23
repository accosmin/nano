#ifndef NANOCV_CONV_LAYER_H
#define NANOCV_CONV_LAYER_H

#include "layer.h"

namespace ncv
{
//        ///
//        /// \brief convolution layer
//        ///
//        class conv_layer_t : public layer_t
//        {
//        public:

//                // constructor
//                conv_layer_t(const string_t& params = string_t());

//                NCV_MAKE_CLONABLE(conv_layer_t, layer_t,
//                                  "convolution layer, parameters: dims=16[1,256],rows=8[1,32],cols=8[1,32]")

//                // resize to process new tensors of the given type
//                virtual size_t resize(const tensor_t& tensor);

//                // reset parameters
//                virtual void zero_params();
//                virtual void random_params(scalar_t min, scalar_t max);

//                // serialize parameters & gradients
//                virtual ovectorizer_t& save_params(ovectorizer_t& s) const;
//                virtual ovectorizer_t& save_grad(ovectorizer_t& s) const;
//                virtual ivectorizer_t& load_params(ivectorizer_t& s);

//                // process inputs (compute outputs & gradients)
//                virtual const tensor_t& forward(const tensor_t& input);
//                virtual const tensor_t& backward(const tensor_t& gradient);

//                // save/load parameters to/from file
//                virtual bool save(boost::archive::binary_oarchive& oa) const;
//                virtual bool load(boost::archive::binary_iarchive& ia);

//                // access functions
//                virtual const tensor_t& input() const { return m_idata; }
//                virtual const tensor_t& output() const { return m_odata; }

//        private:

//                /////////////////////////////////////////////////////////////////////////////////////////

//                size_t idims() const { return m_idata.dim1(); }
//                size_t odims() const { return m_odata.dim1(); }

//                scalar_t bias(size_t o) const { return m_bdata(o, 0, 0); }
//                scalar_t weight(size_t o, size_t i) const { return m_wdata(o, i, 0); }

//                scalar_t& gweight(size_t o, size_t i) const { return m_gwdata(o, i, 0); }
//                scalar_t& gbias(size_t o) const { return m_gbdata(o, 0, 0); }

//                size_t n_kdims() const { return m_kdata.n_dim1(); }
//                size_t n_krows() const { return m_kdata.n_rows(); }
//                size_t n_kcols() const { return m_kdata.n_cols(); }

//                /////////////////////////////////////////////////////////////////////////////////////////

//        private:

//                // attributes
//                string_t                m_params;

//                tensor_t                m_idata;        ///< input buffer
//                tensor_t                m_odata;        // output buffer
//                tensor_t                m_xdata;        // output convolution buffer

//                tensor3d_t              m_kdata;        // convolution/kernel matrices (output)
//                tensor3d_t              m_wdata;        // weights (output, input)
//                tensor3d_t              m_bdata;        // biases (output)

//                mutable tensor3d_t      m_gkdata;       // cumulated gradients
//                mutable tensor3d_t      m_gwdata;
//                mutable tensor3d_t      m_gbdata;
//        };
}

#endif // NANOCV_CONV_LAYER_H

