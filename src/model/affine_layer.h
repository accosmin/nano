#ifndef NANOCV_AFFINE_LAYER_H
#define NANOCV_AFFINE_LAYER_H

//#include "core/tensor3d.h"
//#include "core/tensor4d.h"
//#include "activation/activation.h"

//namespace ncv
//{
//        class affine_layer_t;
//        typedef std::vector<affine_layer_t>             affine_network_t;

//        class affine_layer_param_t;
//        typedef std::vector<affine_layer_param_t>       affine_network_params_t;

//        /////////////////////////////////////////////////////////////////////////////////////////
//        // affine layer:
//        //      - process a set of inputs of size (irows, icols) and produces a set of outputs
//        //              of fixed size using fully connected affine transformations.
//        /////////////////////////////////////////////////////////////////////////////////////////

//        struct affine_layer_param_t
//        {
//                // constructor
//                affine_layer_param_t(size_t outputs = 0,
//                                   const string_t& activation = string_t())
//                        :       m_outputs(outputs),
//                                m_activation(activation)
//                {
//                }

//                friend class boost::serialization::access;
//                template
//                <
//                        class tarchive
//                >
//                void serialize(tarchive & ar, const unsigned int version)
//                {
//                        ar & m_outputs;
//                        ar & m_activation;
//                }

//                // attributes
//                size_t          m_outputs;              // #outputs
//                string_t        m_activation;           // activation function id
//        };

//        class affine_layer_t
//        {
//        public:

//                // constructor
//                affine_layer_t(size_t inputs = 0, size_t irows = 0, size_t icols = 0,
//                             size_t outputs = 0, size_t crows = 0, size_t ccols = 0,
//                             const string_t& activation = string_t());

//                // resize to new dimensions
//                size_t resize(size_t inputs, size_t irows, size_t icols,
//                              size_t outputs, size_t crows, size_t ccols,
//                              const string_t& activation);

//                // reset parameters
//                void zero();
//                void random(scalar_t min = -0.1, scalar_t max = 0.1);
//                void zero_grad() const;

//                // process inputs (compute outputs & gradients)
//                const tensor3d_t& forward(const tensor3d_t& input) const;
//                const tensor3d_t& backward(const tensor3d_t& gradient) const;

//                // serialize/deserialize data
//                friend serializer_t& operator<<(serializer_t& s, const affine_layer_t& layer);
//                friend deserializer_t& operator>>(deserializer_t& s, affine_layer_t& layer);

//                // build convolution network (returns the total number of parameters)
//                static size_t make_network(
//                        size_t idims, size_t irows, size_t icols,       // input size
//                        const affine_network_params_t& network_params,      //
//                        size_t odims,                                   // output size
//                        affine_network_t& network);

//                // show the structure of a network
//                static void print_network(const affine_network_t& network);

//                // process inputs (compute outputs & gradients) for a network
//                static const tensor3d_t& forward(const tensor3d_t& input, const affine_network_t& network);
//                static void backward(const tensor3d_t& gradient, const affine_network_t& network);

//                // access functions
//                size_t n_inputs() const { return m_idata.n_dim1(); }
//                size_t n_irows() const { return m_idata.n_rows(); }
//                size_t n_icols() const { return m_idata.n_cols(); }

//                size_t n_outputs() const { return m_odata.n_dim1(); }
//                size_t n_orows() const { return m_odata.n_rows(); }
//                size_t n_ocols() const { return m_odata.n_cols(); }

//                const tensor4d_t& gdata() const { return m_gdata; }

//        private:

//                friend class boost::serialization::access;
//                template
//                <
//                        class tarchive
//                >
//                void serialize(tarchive & ar, const unsigned int version)
//                {
//                        ar & m_idata;
//                        ar & m_cdata;
//                        ar & m_gdata;
//                        ar & m_odata;
//                        ar & m_activation;

//                        // TODO: save the activation ID and not the activation function itself
//                }

//        private:

//                // attributes
//                mutable tensor3d_t      m_idata;        // input buffer
//                tensor4d_t              m_cdata;        // convolution matrices
//                mutable tensor4d_t      m_gdata;        // cumulated gradient of the convolution matrices
//                mutable tensor3d_t      m_odata;        // output buffer

//                string_t                m_activation;
//                ractivation_t           m_afunc;        // activation/transfer function
//        };

//        // serialize/deserialize data
//        inline serializer_t& operator<<(serializer_t& s, const affine_layer_t& layer)
//        {
//                return s << layer.m_cdata;
//        }

//        inline deserializer_t& operator>>(deserializer_t& s, affine_layer_t& layer)
//        {
//                return s >> layer.m_cdata;
//        }
//}

#endif // NANOCV_AFFINE_LAYER_H
