#pragma once

#include "layer.h"

namespace nano
{
        ///
        /// \brief activation layer: applies a non-linear scalar function to the each input
        ///
        template
        <
                /// activation value: teval_op(input, output)
                typename teval_op,

                /// & its gradient wrt to input, given the output and the gradient: tgrad_op(gradient, input-output)
                typename tgrad_op

        >
        class activation_layer_t : public layer_t
        {
        public:

                using tactivation = activation_layer_t<teval_op, tgrad_op>;

                // constructor
                explicit activation_layer_t(const string_t& parameters = string_t()) :
                        layer_t(parameters)
                {
                }

                // destructor
                virtual ~activation_layer_t() {}

                // clone
                virtual rlayer_t clone(const string_t& parameters) const final
                {
                        return std::make_unique<tactivation>(parameters);
                }
                virtual rlayer_t clone() const final
                {
                        return std::make_unique<tactivation>(*this);
                }

                // resize to process new tensors of the given type
                virtual tensor_size_t resize(const tensor3d_t& tensor) final
                {
                        return _resize(tensor);
                }

                // reset parameters
                virtual void zero_params() final {}
                virtual void random_params(scalar_t min, scalar_t max) final { NANO_UNUSED2(min, max); }

                // serialize parameters
                virtual scalar_t* save_params(scalar_t* params) const final { return params; }
                virtual const scalar_t* load_params(const scalar_t* params) final { return params; }

                // process inputs (compute outputs & gradients)
                virtual const tensor3d_t& output(const tensor3d_t& input) final { return _output(input); }
                virtual const tensor3d_t& ginput(const tensor3d_t& output) final { return _ginput(output); }
                virtual void gparam(const tensor3d_t& output, scalar_t*) final { return _gparam(output); }

                // access functions
                virtual tensor_size_t idims() const final { return m_data.size<0>(); }
                virtual tensor_size_t irows() const final { return m_data.size<1>(); }
                virtual tensor_size_t icols() const final { return m_data.size<2>(); }
                virtual tensor_size_t odims() const final { return m_data.size<0>(); }
                virtual tensor_size_t orows() const final { return m_data.size<1>(); }
                virtual tensor_size_t ocols() const final { return m_data.size<2>(); }
                virtual tensor_size_t psize() const final { return 0; }
                virtual tensor_size_t flops() const final { return 10 * m_data.size(); }

        private:

                // resize to process new inputs, returns the number of parameters
                tensor_size_t _resize(const tensor3d_t& tensor)
                {
                        m_data.resize(tensor.dims());

                        return 0;
                }

                // output
                const tensor3d_t& _output(const tensor3d_t& input)
                {
                        assert(m_data.size<0>() == input.size<0>());
                        assert(m_data.size<1>() == input.size<1>());
                        assert(m_data.size<2>() == input.size<2>());

                        teval_op()(input.vector(), m_data.vector());

                        return m_data;
                }

                // gradient
                const tensor3d_t& _ginput(const tensor3d_t& output)
                {
                        assert(m_data.size<0>() == output.size<0>());
                        assert(m_data.size<1>() == output.size<1>());
                        assert(m_data.size<2>() == output.size<2>());

                        tgrad_op()(output.vector(), m_data.vector());

                        return m_data;
                }

                // gradient
                void _gparam(const tensor3d_t& output)
                {
                        NANO_UNUSED1_RELEASE(output);

                        assert(m_data.size<0>() == output.size<0>());
                        assert(m_data.size<1>() == output.size<1>());
                        assert(m_data.size<2>() == output.size<2>());
                }

        private:

                // attributes
                tensor3d_t      m_data;         ///< input-output buffer
        };
}

