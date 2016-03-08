#pragma once

#include "cortex/layer.h"
#include "tensor/transform.hpp"

namespace cortex
{
        ///
        /// \brief activation layer: applies a non-linear scalar function to the each input
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
                explicit activation_layer_t(const string_t& parameters)
                        :       layer_t(parameters)
                {
                }

                // destructor
                virtual ~activation_layer_t() {}

                // resize to process new tensors of the given type
                virtual tensor_size_t resize(const tensor_t& tensor) override
                {
                        return _resize(tensor);
                }

                // reset parameters
                virtual void zero_params() override {}
                virtual void random_params(scalar_t min, scalar_t max) override { ZOB_UNUSED2(min, max); }

                // serialize parameters
                virtual scalar_t* save_params(scalar_t* params) const override { return params; }
                virtual const scalar_t* load_params(const scalar_t* params) override { return params; }

                // process inputs (compute outputs & gradients)
                virtual const tensor_t& output(const tensor_t& input) override { return _output(input); }
                virtual const tensor_t& ginput(const tensor_t& output) override { return _ginput(output); }
                virtual void gparam(const tensor_t& output, scalar_t*) override { return _gparam(output); }

                // access functions
                virtual tensor_size_t idims() const override { return m_data.dims(); }
                virtual tensor_size_t irows() const override { return m_data.rows(); }
                virtual tensor_size_t icols() const override { return m_data.cols(); }
                virtual tensor_size_t odims() const override { return m_data.dims(); }
                virtual tensor_size_t orows() const override { return m_data.rows(); }
                virtual tensor_size_t ocols() const override { return m_data.cols(); }
                virtual tensor_size_t psize() const override { return 0; }

        private:

                // resize to process new inputs, returns the number of parameters
                tensor_size_t _resize(const tensor_t& tensor)
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

                        tensor::transform(input, m_data,
                                          [op = teval_op()] (auto x) { return op(x); });

                        return m_data;
                }

                // gradient
                const tensor_t& _ginput(const tensor_t& output)
                {
                        assert(m_data.dims() == output.dims());
                        assert(m_data.rows() == output.rows());
                        assert(m_data.cols() == output.cols());

                        tensor::transform(output, m_data, m_data,
                                          [op = tgrad_op()] (auto g, auto o) { return op(g, o); });

                        return m_data;
                }

                // gradient
                void _gparam(const tensor_t& output)
                {
                        ZOB_UNUSED1_RELEASE(output);

                        assert(m_data.dims() == output.dims());
                        assert(m_data.rows() == output.rows());
                        assert(m_data.cols() == output.cols());
                }

        private:

                // attributes
                tensor_t        m_data;         ///< input-output buffer
        };
}

