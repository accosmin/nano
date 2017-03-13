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
                virtual bool configure(const dim3d_t& idims) override;
                virtual bool configure(const tensor3d_map_t idata, const tensor3d_map_t odata, const vector_map_t) override;

                virtual void output() override;
                virtual void ginput() override;
                virtual void gparam() override;

                virtual dim3d_t idims() const override { return m_idata.dims(); }
                virtual dim3d_t odims() const override { return m_odata.dims(); }
                virtual tensor_size_t psize() const override { return 0; }
                virtual tensor_size_t flops() const override { return 10 * m_idata.size(); }

        private:

                // attributes
                tensor3d_map_t  m_idata;
                tensor3d_map_t  m_odata;
        };

        template <typename top>
        rlayer_t activation_layer_t<top>::clone() const
        {
                return std::make_unique<activation_layer_t<top>>(*this);
        }

        template <typename top>
        bool activation_layer_t<top>::configure(const dim3d_t& idims)
        {
                return true;
        }

        template <typename top>
        bool activation_layer_t<top>::configure(const tensor3d_map_t idata, const tensor3d_map_t odata, const vector_map_t)
        {
                m_idata = idata;
                m_odata = odata;
                return true;
        }

        template <typename top>
        void activation_layer_t<top>::output()
        {
                m_odata.vector() = top::output(m_idata.vector());
        }

        template <typename top>
        void activation_layer_t<top>::ginput()
        {
                m_idata.vector() = m_odata.array() * top::ginput(m_idata.vector(), m_odata.vector());
        }

        template <typename top>
        void activation_layer_t<top>::gparam()
        {
        }
}
