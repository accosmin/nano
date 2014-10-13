#pragma once

#include "layer.h"
#include "tensor/transform.hpp"

namespace ncv
{
        ///
        /// activation layer: applies a non-linear scalar function to the each input
        ///
        template
        <
                /// activation value o: o = teval_op(x)
                typename teval_op,

                /// & its gradient wrt to input x, given the output o and propagated gradient g: g = tgrad_op(g, o)
                typename tgrad_op

        >
        class activation_layer_t : public layer_t
        {
        public:

                // constructor
                activation_layer_t(const string_t& parameters)
                        :       layer_t(parameters)
                {
                }

                // destructor
                virtual ~activation_layer_t() {}

                // resize to process new tensors of the given type
                virtual size_t resize(const tensor_t& tensor)
                {
                        return _resize(tensor);
                }

                // reset parameters
                virtual void zero_params() {}
                virtual void random_params(scalar_t min, scalar_t max) {}

                // serialize parameters
                virtual scalar_t* save_params(scalar_t* params) const { return params; }
                virtual const scalar_t* load_params(const scalar_t* params) { return params; }

                // serialize parameters (to disk)
                virtual boost::archive::binary_oarchive& save(boost::archive::binary_oarchive& oa) const { return oa; }
                virtual boost::archive::binary_iarchive& load(boost::archive::binary_iarchive& ia) { return ia; }

                // process inputs (compute outputs & gradients)
                virtual const tensor_t& output(const tensor_t& input) { return _output(input); }
                virtual const tensor_t& igrad(const tensor_t& output) { return _igrad(output); }
                virtual void pgrad(const tensor_t& output, scalar_t*) { return _pgrad(output); }

                // access functions
                virtual size_t idims() const { return m_data.dims(); }
                virtual size_t irows() const { return m_data.rows(); }
                virtual size_t icols() const { return m_data.cols(); }
                virtual size_t odims() const { return m_data.dims(); }
                virtual size_t orows() const { return m_data.rows(); }
                virtual size_t ocols() const { return m_data.cols(); }
                virtual size_t psize() const { return 0; }

        private:

                // resize to process new inputs, returns the number of parameters
                size_t _resize(const tensor_t& tensor)
                {
                        m_data.resize(tensor.dims(), tensor.rows(), tensor.cols());

                        return 0;
                }

                // output
                const tensor_t& _output(const tensor_t& input)
                {
                        assert(m_data.dims() == input.dims());
                        assert(m_data.rows() == input.rows());
                        assert(m_data.cols() == input.cols());

                        tensor::transform(input, m_data, std::bind(teval_op(), _1));

                        return m_data;
                }

                // gradient
                const tensor_t& _igrad(const tensor_t& output)
                {
                        assert(m_data.dims() == output.dims());
                        assert(m_data.rows() == output.rows());
                        assert(m_data.cols() == output.cols());

                        tensor::transform(output, m_data, m_data, std::bind(tgrad_op(), _1, _2));

                        return m_data;
                }

                // gradient
                void _pgrad(const tensor_t& output)
                {
                        assert(m_data.dims() == output.dims());
                        assert(m_data.rows() == output.rows());
                        assert(m_data.cols() == output.cols());
                }

        private:

                // attributes
                tensor_t                m_data;         ///< input-output buffer
        };
}

