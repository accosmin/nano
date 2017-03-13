#pragma once

#include "layer.h"

namespace nano
{
        ///
        /// \brief activation layer: applies a non-linear scalar function to the each input
        ///
        template <typename top>
        struct activation_layer_t final : public layer_t
        {
                explicit activation_layer_t(const string_t& parameters = string_t()) : layer_t(parameters) {}

                virtual rlayer_t clone() const override;

                virtual tensor_size_t resize(const tensor3d_t& tensor) override;

                virtual void random(const scalar_t, const scalar_t) override {}
                virtual scalar_t* save_params(scalar_t* params) const override { return params; }
                virtual const scalar_t* load_params(const scalar_t* params) override { return params; }

                virtual bool save(obstream_t&) const override { return true; }
                virtual bool load(ibstream_t&) override { return true; }

                virtual const tensor3d_t& output(const tensor3d_t& input) override;
                virtual const tensor3d_t& ginput(const tensor3d_t& output) override;
                virtual void gparam(const tensor3d_t& output, scalar_t*) override;

                virtual dim3d_t idims() const override { return m_idata.dims(); }
                virtual dim3d_t odims() const override { return m_odata.dims(); }
                virtual tensor_size_t psize() const override { return 0; }
                virtual tensor_size_t flops() const override { return 10 * m_idata.size(); }

        private:

                // attributes
                tensor3d_t      m_idata;
                tensor3d_t      m_odata;
        };

        template <typename top>
        rlayer_t activation_layer_t<top>::clone() const
        {
                return std::make_unique<activation_layer_t<top>>(*this);
        }

        template <typename top>
        tensor_size_t activation_layer_t<top>::resize(const tensor3d_t& tensor)
        {
                m_idata.resize(tensor.dims());
                m_odata.resize(tensor.dims());
                return 0;
        }

        template <typename top>
        const tensor3d_t& activation_layer_t<top>::output(const tensor3d_t& input)
        {
                assert(m_idata.dims() == input.dims());

                m_idata.vector() = input.vector();
                m_odata.vector() = top::output(m_idata.vector());

                return m_odata;
        }

        template <typename top>
        const tensor3d_t& activation_layer_t<top>::ginput(const tensor3d_t& output)
        {
                assert(m_odata.dims() == output.dims());

                m_idata.vector() = output.array() * top::ginput(m_idata.vector(), m_odata.vector());

                return m_idata;
        }

        template <typename top>
        void activation_layer_t<top>::gparam(const tensor3d_t& output, scalar_t*)
        {
                assert(m_odata.dims() == output.dims());

                NANO_UNUSED1_RELEASE(output);
        }
}
