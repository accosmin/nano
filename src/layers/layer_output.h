#ifndef NANOCV_OUTPUT_LAYER_H
#define NANOCV_OUTPUT_LAYER_H

#include "layer.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // fully-connected output layer:
        //
        // parameters:
        //      odims=10[1,256]         - number of outputs
        /////////////////////////////////////////////////////////////////////////////////////////

        class output_layer_t : public layer_t
        {
        public:

                // constructor
                explicit output_layer_t(const string_t& params = string_t());

                NCV_MAKE_CLONABLE(output_layer_t, layer_t, "fully-connected output layer, parameters: odims=10")

                // resize to process new inputs, returns the number of parameters
                virtual size_t resize(size_t idims, size_t irows, size_t icols);

                // reset parameters
                virtual void zero_params();
                virtual void random_params(scalar_t min, scalar_t max);

                // serialize parameters & gradients
                virtual ovectorizer_t& save_params(ovectorizer_t& s) const;
                virtual ovectorizer_t& save_grad(ovectorizer_t& s) const;
                virtual ivectorizer_t& load_params(ivectorizer_t& s);

                // process inputs (compute outputs & gradients)
                virtual const tensor3d_t& forward(const tensor3d_t& input) const;
                virtual const tensor3d_t& backward(const tensor3d_t& gradient) const;

                // save/load parameters to/from file
                virtual bool save(boost::archive::binary_oarchive& oa) const;
                virtual bool load(boost::archive::binary_iarchive& ia);

                // save layer description as image
                virtual bool save_as_image(const string_t&) const { return true; }

                // access functions
                virtual size_t n_idims() const { return m_idata.n_dim1(); }
                virtual size_t n_irows() const { return m_idata.n_rows(); }
                virtual size_t n_icols() const { return m_idata.n_cols(); }

                virtual size_t n_odims() const { return m_odata.n_dim1(); }
                virtual size_t n_orows() const { return m_odata.n_rows(); }
                virtual size_t n_ocols() const { return m_odata.n_cols(); }

        private:

                scalar_t bias(size_t o) const { return m_bdata(o, 0, 0); }
                scalar_t& gbias(size_t o) const { return m_gbdata(o, 0, 0); }

        private:

                // attributes
                string_t                m_params;

                mutable tensor3d_t      m_idata;        // input buffer
                mutable tensor3d_t      m_odata;        // output buffer

                tensor4d_t              m_kdata;        // fully-connected matrices
                tensor3d_t              m_bdata;        // biases (output)

                mutable tensor4d_t      m_gkdata;       // cumulated gradients
                mutable tensor3d_t      m_gbdata;
        };
}

#endif // NANOCV_OUTPUT_LAYER_H
