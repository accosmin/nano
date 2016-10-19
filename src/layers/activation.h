#pragma once

#include "layer.h"

namespace nano
{
        ///
        /// \brief activation layer: applies a non-linear scalar function to the each input
        ///
        template
        <
                typename teval_op,      ///< activation value: teval_op(input, output)
                typename tgrad_op       ///< gradient wrt to input: tgrad_op(grad, input-output)
        >
        class activation_layer_t : public layer_t
        {
        public:

                using tactivation = activation_layer_t<teval_op, tgrad_op>;

                // constructor
                explicit activation_layer_t(const string_t& parameters = string_t()) : layer_t(parameters) {}

                // destructor
                virtual ~activation_layer_t() {}

                // clone
                virtual rlayer_t clone(const string_t& parameters) const override final
                {
                        return std::make_unique<tactivation>(parameters);
                }
                virtual rlayer_t clone() const override final
                {
                        return std::make_unique<tactivation>(*this);
                }

                // resize to process new tensors of the given type
                virtual tensor_size_t resize(const tensor3d_t& tensor) override final;

                // reset parameters
                virtual void zero_params() override final {}
                virtual void random_params(scalar_t, scalar_t) override final { }

                // serialize parameters
                virtual scalar_t* save_params(scalar_t* params) const override final { return params; }
                virtual const scalar_t* load_params(const scalar_t* params) override final { return params; }

                // process inputs (compute outputs & gradients)
                virtual const tensor3d_t& output(const tensor3d_t& input) override final;
                virtual const tensor3d_t& ginput(const tensor3d_t& output) override final;
                virtual void gparam(const tensor3d_t& output, scalar_t*) override final;

                // access functions
                virtual tensor_size_t idims() const override final { return m_data.size<0>(); }
                virtual tensor_size_t irows() const override final { return m_data.size<1>(); }
                virtual tensor_size_t icols() const override final { return m_data.size<2>(); }
                virtual tensor_size_t odims() const override final { return m_data.size<0>(); }
                virtual tensor_size_t orows() const override final { return m_data.size<1>(); }
                virtual tensor_size_t ocols() const override final { return m_data.size<2>(); }
                virtual tensor_size_t psize() const override final { return 0; }
                virtual tensor_size_t flops() const override final { return 10 * m_data.size(); }

        private:

                // attributes
                tensor3d_t      m_data;         ///< input-output buffer
        };

        template <typename teval_op, typename tgrad_op>
        tensor_size_t activation_layer_t<teval_op, tgrad_op>::resize(const tensor3d_t& tensor)
        {
                m_data.resize(tensor.dims());
                return 0;
        }

        template <typename teval_op, typename tgrad_op>
        const tensor3d_t& activation_layer_t<teval_op, tgrad_op>::output(const tensor3d_t& input)
        {
                assert(m_data.size<0>() == input.size<0>());
                assert(m_data.size<1>() == input.size<1>());
                assert(m_data.size<2>() == input.size<2>());

                teval_op()(input.vector(), m_data.vector());

                return m_data;
        }

        template <typename teval_op, typename tgrad_op>
        const tensor3d_t& activation_layer_t<teval_op, tgrad_op>::ginput(const tensor3d_t& output)
        {
                assert(m_data.size<0>() == output.size<0>());
                assert(m_data.size<1>() == output.size<1>());
                assert(m_data.size<2>() == output.size<2>());

                tgrad_op()(output.vector(), m_data.vector());

                return m_data;
        }

        template <typename teval_op, typename tgrad_op>
        void activation_layer_t<teval_op, tgrad_op>::gparam(const tensor3d_t& output, scalar_t*)
        {
                NANO_UNUSED1_RELEASE(output);

                assert(m_data.size<0>() == output.size<0>());
                assert(m_data.size<1>() == output.size<1>());
                assert(m_data.size<2>() == output.size<2>());
        }
}

