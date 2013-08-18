#ifndef NANOCV_ACTIVATION_LAYER_H
#define NANOCV_ACTIVATION_LAYER_H

#include "layer.h"
#include "core/tensor3d.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // activation layer:
        //      applies a non-linear scalar function to the each input.
        /////////////////////////////////////////////////////////////////////////////////////////

        class activation_layer_t : public layer_t
        {
        public:

                // destructor
                virtual ~activation_layer_t() {}

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
                virtual size_t n_idims() const { return m_data.n_dim1(); }
                virtual size_t n_irows() const { return m_data.n_rows(); }
                virtual size_t n_icols() const { return m_data.n_cols(); }

                virtual size_t n_odims() const { return m_data.n_dim1(); }
                virtual size_t n_orows() const { return m_data.n_rows(); }
                virtual size_t n_ocols() const { return m_data.n_cols(); }

        protected:

                // activation outputs & gradients
                virtual scalar_t value(scalar_t ix) const = 0;
                virtual scalar_t vgrad(scalar_t gx, scalar_t ox) const = 0;

        private:

                // attributes
                mutable tensor3d_t      m_data;         // input-output buffer
        };
}

#endif // NANOCV_ACTIVATION_LAYER_H
