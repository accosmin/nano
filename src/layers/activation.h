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
                explicit activation_layer_t(const string_t& parameters = string_t()) :
                        layer_t(parameters),
                        m_idims({0, 0, 0}),
                        m_odims({0, 0, 0})
                {
                }

                virtual rlayer_t clone() const override;
                virtual void configure(const dim3d_t&) override;
                virtual void output(tensor3d_map_t, tensor1d_map_t, tensor3d_map_t) override;
                virtual void ginput(tensor3d_map_t, tensor1d_map_t, tensor3d_map_t) override;
                virtual void gparam(tensor3d_map_t, tensor1d_map_t, tensor3d_map_t) override;

                virtual dim3d_t idims() const override { return m_idims; }
                virtual dim3d_t odims() const override { return m_odims; }
                virtual tensor_size_t psize() const override { return 0; }
                virtual tensor_size_t flops() const override { return 10 * nano::size(m_idims); }

        private:

                // attributes
                dim3d_t         m_idims;
                dim3d_t         m_odims;
        };

        template <typename top>
        rlayer_t activation_layer_t<top>::clone() const
        {
                return std::make_unique<activation_layer_t<top>>(*this);
        }

        template <typename top>
        void activation_layer_t<top>::configure(const dim3d_t& idims)
        {
                m_idims = idims;
                m_odims = idims;
        }

        template <typename top>
        void activation_layer_t<top>::output(tensor3d_map_t idata, tensor1d_map_t param, tensor3d_map_t odata)
        {
                assert(idata.dims() == idims());
                assert(param.size() == psize());
                assert(odata.dims() == odims());
                NANO_UNUSED1_RELEASE(param);

                odata.array() = top::output(idata.array());
        }

        template <typename top>
        void activation_layer_t<top>::ginput(tensor3d_map_t idata, tensor1d_map_t param, tensor3d_map_t odata)
        {
                assert(idata.dims() == idims());
                assert(param.size() == psize());
                assert(odata.dims() == odims());
                NANO_UNUSED1_RELEASE(param);

                idata.array() = odata.array() * top::ginput(idata.array());
        }

        template <typename top>
        void activation_layer_t<top>::gparam(tensor3d_map_t idata, tensor1d_map_t param, tensor3d_map_t odata)
        {
                assert(idata.dims() == idims());
                assert(param.size() == psize());
                assert(odata.dims() == odims());
                NANO_UNUSED3_RELEASE(idata, param, odata);
        }
}
