#ifndef NANOCV_CONV_LAYER_H
#define NANOCV_CONV_LAYER_H

#include "layer.h"
#include "core/tensor3d.h"
#include "core/tensor4d.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // convolution layer:
        //
        // parameters:
        //      convs=16[1,256]         - number of convolutions
        //      crows=8[1,256]          - convolution size
        //      ccols=8[1,256]          - convolution size
        /////////////////////////////////////////////////////////////////////////////////////////

        class conv_layer_t : public layer_t
        {
        public:

                // constructor
                conv_layer_t(const string_t& params = string_t());

                NCV_MAKE_CLONABLE(conv_layer_t, layer_t, "convolution layer, parameters: convs=16,crows=8,ccols=8")

                // resize to process new inputs, returns the number of parameters
                virtual size_t resize(size_t idims, size_t irows, size_t icols);

                // reset parameters
                virtual void zero_params();
                virtual void random_params(scalar_t min, scalar_t max);
                virtual void zero_grad() const;

                // serialize parameters & gradients
                virtual serializer_t& save_params(serializer_t& s) const;
                virtual serializer_t& save_grad(serializer_t& s) const;
                virtual deserializer_t& load_params(deserializer_t& s);

                // process inputs (compute outputs & gradients)
                virtual const tensor3d_t& forward(const tensor3d_t& input) const;
                virtual const tensor3d_t& backward(const tensor3d_t& gradient) const;

                // save/load parameters to/from file
                virtual bool save(boost::archive::binary_oarchive& oa) const;
                virtual bool load(boost::archive::binary_iarchive& ia);

                // access functions
                virtual size_t n_idims() const { return m_idata.n_dim1(); }
                virtual size_t n_irows() const { return m_idata.n_rows(); }
                virtual size_t n_icols() const { return m_idata.n_cols(); }

                virtual size_t n_odims() const { return m_odata.n_dim1(); }
                virtual size_t n_orows() const { return m_odata.n_rows(); }
                virtual size_t n_ocols() const { return m_odata.n_cols(); }

                size_t n_kdim1() const { return m_kdata.n_dim1(); }
                size_t n_kdim2() const { return m_kdata.n_dim2(); }
                size_t n_krows() const { return m_kdata.n_rows(); }
                size_t n_kcols() const { return m_kdata.n_cols(); }

        private:

                // attributes
                string_t                m_params;

                mutable tensor3d_t      m_idata;        // input buffer
                tensor4d_t              m_kdata;        // convolution/kernel matrices
                mutable tensor4d_t      m_gdata;        // cumulated gradient of the convolution matrices
                mutable tensor3d_t      m_odata;        // output buffer
        };
}

#endif // NANOCV_CONV_LAYER_H
