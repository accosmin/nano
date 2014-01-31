#ifndef NANOCV_SOFTMAX_POOL_LAYER_H
#define NANOCV_SOFTMAX_POOL_LAYER_H

#include "layer.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // softmax pooling layer:
        //      down-sample by 2 from a 3x3 neighbouring region using a soft-max weighting.
        /////////////////////////////////////////////////////////////////////////////////////////

        class softmax_pool_layer_t : public layer_t
        {
        public:

                // constructor
                softmax_pool_layer_t(const string_t& params = string_t());

                NCV_MAKE_CLONABLE(softmax_pool_layer_t, layer_t,
                                  "soft-max pooling layer")

                // resize to process new inputs, returns the number of parameters
                virtual size_t resize(size_t idims, size_t irows, size_t icols);

                // reset parameters
                virtual void zero_params() {}
                virtual void random_params(scalar_t min, scalar_t max) {}

                // serialize parameters & gradients
                virtual ovectorizer_t& save_params(ovectorizer_t& s) const { return s; }
                virtual ovectorizer_t& save_grad(ovectorizer_t& s) const { return s; }
                virtual ivectorizer_t& load_params(ivectorizer_t& s) { return s; }

                // process inputs (compute outputs & gradients)
                virtual const tensor3d_t& forward(const tensor3d_t& input) const;
                virtual const tensor3d_t& backward(const tensor3d_t& gradient) const;

                // save/load parameters to/from file
                virtual bool save(boost::archive::binary_oarchive& oa) const { return true; }
                virtual bool load(boost::archive::binary_iarchive& ia) { return true; }

                // save layer description as image
                virtual bool save_as_image(const string_t& basepath) const { return true; }

                // access functions
                virtual size_t n_idims() const { return m_idata.n_dim1(); }
                virtual size_t n_irows() const { return m_idata.n_rows(); }
                virtual size_t n_icols() const { return m_idata.n_cols(); }

                virtual size_t n_odims() const { return m_odata.n_dim1(); }
                virtual size_t n_orows() const { return m_odata.n_rows(); }
                virtual size_t n_ocols() const { return m_odata.n_cols(); }

        private:

                // attributes
                mutable tensor3d_t      m_idata;        // input buffer
                mutable tensor3d_t      m_odata;        // output buffer

                mutable tensor3d_t      m_wdata;        // pooling weights
                mutable tensor3d_t      m_sdata;
                mutable tensor3d_t      m_tdata;
        };
}

#endif // NANOCV_SOFTMAX_POOL_LAYER_H

