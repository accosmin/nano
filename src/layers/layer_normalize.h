#pragma once

#include "layer.h"

namespace nano
{
        ///
        /// \brief normalization type.
        ///
        enum class norm_type
        {
                global,                 ///< globally using all feature planes
                plane,                  ///< per feature plane
        };

        template <>
        inline enum_map_t<norm_type> enum_string<norm_type>()
        {
                return
                {
                        { norm_type::global,    "global" },
                        { norm_type::plane,     "plane" }
                };
        }

        ///
        /// \brief normalize layer: transforms the input tensor to have zero mean and one variance,
        ///     either globally or for each feature map.
        ///
        struct normalize_layer_t final : public layer_t
        {
                explicit normalize_layer_t(const string_t& params = string_t());

                rlayer_t clone() const override;
                bool configure(const tensor3d_dims_t& idims, const string_t& name) override;
                void output(const tensor4d_t& idata, const tensor1d_t& pdata, tensor4d_t& odata) override;
                void ginput(tensor4d_t& idata, const tensor1d_t& pdata, const tensor4d_t& odata) override;
                void gparam(const tensor4d_t& idata, tensor1d_t& pdata, const tensor4d_t& odata) override;

                tensor_size_t fanin() const override { return 1; }
                tensor3d_dims_t idims() const override { return m_xdims; }
                tensor3d_dims_t odims() const override { return m_xdims; }
                tensor1d_dims_t pdims() const override { return {0}; }

                const probe_t& probe_output() const override { return m_probe_output; }
                const probe_t& probe_ginput() const override { return m_probe_ginput; }
                const probe_t& probe_gparam() const override { return m_probe_gparam; }

        private:

                // attributes
                tensor3d_dims_t m_xdims;        ///< input/output dimensions
                probe_t         m_probe_output;
                probe_t         m_probe_ginput;
                probe_t         m_probe_gparam;
        };
}
