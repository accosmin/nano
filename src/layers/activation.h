#pragma once

#include "layer.h"

namespace nano
{
        ///
        /// \brief activation layer: applies a non-linear scalar function to the each input
        ///
        template <typename top>
        struct activation_layer_t : public layer_t
        {
                explicit activation_layer_t(const string_t& parameters = string_t()) : layer_t(parameters) {}
                virtual ~activation_layer_t() {}

                virtual rlayer_t clone(const string_t& parameters) const override final;
                virtual rlayer_t clone() const override final;

                virtual tensor_size_t resize(const tensor3d_t& tensor) override final;

                virtual void zero_params() override final {}
                virtual void random_params(scalar_t, scalar_t) override final { }

                virtual scalar_t* save_params(scalar_t* params) const override final { return params; }
                virtual const scalar_t* load_params(const scalar_t* params) override final { return params; }

                virtual const tensor3d_t& output(const tensor3d_t& input) override final;
                virtual const tensor3d_t& ginput(const tensor3d_t& output) override final;
                virtual void gparam(const tensor3d_t& output, scalar_t*) override final;

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

        template <typename top>
        rlayer_t activation_layer_t<top>::clone(const string_t& parameters) const
        {
                return std::make_unique<activation_layer_t<top>>(parameters);
        }

        template <typename top>
        rlayer_t activation_layer_t<top>::clone() const
        {
                return std::make_unique<activation_layer_t<top>>(*this);
        }

        template <typename top>
        tensor_size_t activation_layer_t<top>::resize(const tensor3d_t& tensor)
        {
                m_data.resize(tensor.dims());
                return 0;
        }

        template <typename top>
        const tensor3d_t& activation_layer_t<top>::output(const tensor3d_t& input)
        {
                assert(m_data.size<0>() == input.size<0>());
                assert(m_data.size<1>() == input.size<1>());
                assert(m_data.size<2>() == input.size<2>());

                top::output(input.vector(), m_data.vector());

                return m_data;
        }

        template <typename top>
        const tensor3d_t& activation_layer_t<top>::ginput(const tensor3d_t& output)
        {
                assert(m_data.size<0>() == output.size<0>());
                assert(m_data.size<1>() == output.size<1>());
                assert(m_data.size<2>() == output.size<2>());

                top::ginput(output.vector(), m_data.vector());

                return m_data;
        }

        template <typename top>
        void activation_layer_t<top>::gparam(const tensor3d_t& output, scalar_t*)
        {
                NANO_UNUSED1_RELEASE(output);

                assert(m_data.size<0>() == output.size<0>());
                assert(m_data.size<1>() == output.size<1>());
                assert(m_data.size<2>() == output.size<2>());
        }
}

