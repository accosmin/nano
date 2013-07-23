#ifndef NANOCV_CONV_LAYER_H
#define NANOCV_CONV_LAYER_H

#include "core/tensor3d.h"
#include "core/tensor4d.h"
#include "activation/activation.h"

namespace ncv
{
        class conv_layer_t;
        typedef std::vector<conv_layer_t>               conv_layers_t;

        class conv_layer_param_t;
        typedef std::vector<conv_layer_param_t>         conv_layer_params_t;

        /////////////////////////////////////////////////////////////////////////////////////////
        // convolution layer:
        //      - process a set of inputs of size (irows, icols) and produces
        //              a set of outputs using convolution matrices of size (crows, ccols).
        /////////////////////////////////////////////////////////////////////////////////////////

        struct conv_layer_param_t
        {
                // constructor
                conv_layer_param_t(size_t convs = 0, size_t crows = 0, size_t ccols = 0,
                                   const string_t& activation = string_t())
                        :       m_convs(convs), m_crows(crows), m_ccols(ccols),
                                m_activation(activation)
                {
                }

                friend class boost::serialization::access;
                template
                <
                        class tarchive
                >
                void serialize(tarchive & ar, const unsigned int version)
                {
                        ar & m_convs;
                        ar & m_crows;
                        ar & m_ccols;
                        ar & m_activation;
                }

                // attributes
                size_t          m_convs;                // #convolutions
                size_t          m_crows;                // convolution size (rows)
                size_t          m_ccols;                // convolution size (columns)
                string_t        m_activation;           // activation function id
        };

        class conv_layer_t
        {
        public:

                // constructor
                conv_layer_t(size_t inputs = 0, size_t irows = 0, size_t icols = 0,
                             size_t outputs = 0, size_t crows = 0, size_t ccols = 0,
                             const string_t& activation = string_t());

                // resize to new dimensions
                size_t resize(size_t inputs, size_t irows, size_t icols,
                              size_t outputs, size_t crows, size_t ccols,
                              const string_t& activation);

                // reset parameters
                void zero_params();
                void random_params(scalar_t min = -0.1, scalar_t max = 0.1);
                void zero_grad() const;

                // process inputs (compute outputs & gradients)
                const tensor3d_t& forward(const tensor3d_t& input) const;
                const tensor3d_t& backward(const tensor3d_t& gradient) const;

                // serialize/deserialize data
                friend serializer_t& operator<<(serializer_t& s, const conv_layer_t& layer);
                friend deserializer_t& operator>>(deserializer_t& s, conv_layer_t& layer);

                // access functions
                size_t n_inputs() const { return m_idata.n_dim1(); }
                size_t n_irows() const { return m_idata.n_rows(); }
                size_t n_icols() const { return m_idata.n_cols(); }

                size_t n_outputs() const { return m_odata.n_dim1(); }
                size_t n_orows() const { return m_odata.n_rows(); }
                size_t n_ocols() const { return m_odata.n_cols(); }

                const tensor4d_t& kdata() const { return m_kdata; }
                const tensor4d_t& gdata() const { return m_gdata; }

                const string_t& activation() const { return m_activation; }

        private:

                friend class boost::serialization::access;
                template
                <
                        class tarchive
                >
                void serialize(tarchive & ar, const unsigned int version)
                {
                        ar & m_idata;
                        ar & m_kdata;
                        ar & m_gdata;
                        ar & m_odata;
                        ar & m_activation;

                        set_activation();
                }

                // setup the activation function
                void set_activation();

        private:

                // attributes
                mutable tensor3d_t      m_idata;        // input buffer
                tensor4d_t              m_kdata;        // convolution/kernel matrices
                mutable tensor4d_t      m_gdata;        // cumulated gradient of the convolution matrices
                mutable tensor3d_t      m_odata;        // output buffer

                string_t                m_activation;
                ractivation_t           m_afunc;        // activation/transfer function
        };

        // serialize/deserialize data
        inline serializer_t& operator<<(serializer_t& s, const conv_layer_t& layer)
        {
                return s << layer.m_kdata;
        }

        inline deserializer_t& operator>>(deserializer_t& s, conv_layer_t& layer)
        {
                return s >> layer.m_kdata;
        }
}

#endif // NANOCV_CONV_LAYER_H
