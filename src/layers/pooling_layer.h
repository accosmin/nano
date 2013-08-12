#ifndef NANOCV_POOLING_LAYER_H
#define NANOCV_POOLING_LAYER_H

#include "layer.h"
#include "core/tensor3d.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // pooling layer:
        //      transforms a set of fixed size outputs to a single output of the same size.
        /////////////////////////////////////////////////////////////////////////////////////////

        class pooling_layer_t : public layer_t
        {
        public:

                // destructor
                virtual ~pooling_layer_t() {}

                // resize to process new inputs, returns the number of parameters
                virtual size_t resize(size_t idims, size_t irows, size_t icols);

                // reset parameters
                virtual void zero_params();
                virtual void random_params(scalar_t min, scalar_t max);
                virtual void zero_grad() const;

                // serialize parameters & gradients
                virtual serializer_t& save_params(serializer_t& s) const;
                virtual serializer_t& save_grad(serializer_t& s) const;
                virtual deserializer_t& load_params(deserializer_t& s) const;

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

        protected:

                // pool outputs & gradients
                virtual scalar_t forward_pool(scalar_t ox, scalar_t ix) const = 0;
                virtual scalar_t backward_pool(scalar_t gx, scalar_t ox, scalar_t ix) const = 0;

        private:

                // attributes
                mutable tensor3d_t      m_idata;        // input buffer
                mutable tensor3d_t      m_odata;        // output buffer
        };
}

#endif // NANOCV_POOLING_LAYER_H
