#pragma once

#include "layer.h"
#include "text/enum_string.h"

namespace nano
{
        enum class norm_type
        {
                global,                 ///< globablly using all feature planes
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
        struct normalize_layer_t : public layer_t
        {
                explicit normalize_layer_t(const string_t& params = string_t());

                virtual rlayer_t clone() const override;
                virtual void configure(const tensor3d_dims_t&, const string_t&) override;
                virtual void output(tensor3d_const_map_t, tensor1d_const_map_t, tensor3d_map_t) override;
                virtual void ginput(tensor3d_map_t, tensor1d_const_map_t, tensor3d_const_map_t) override;
                virtual void gparam(tensor3d_const_map_t, tensor1d_map_t, tensor3d_const_map_t) override;

                virtual tensor3d_dims_t idims() const override { return m_idims; }
                virtual tensor3d_dims_t odims() const override { return m_odims; }
                virtual tensor_size_t fanin() const override { return 1; }
                virtual tensor_size_t psize() const override { return 0; }
                virtual const probe_t& probe_output() const override { return m_probe_output; }
                virtual const probe_t& probe_ginput() const override { return m_probe_ginput; }
                virtual const probe_t& probe_gparam() const override { return m_probe_gparam; }

        private:

                // attributes
                tensor3d_dims_t m_idims;
                tensor3d_dims_t m_odims;
                norm_type       m_type;
                probe_t         m_probe_output;
                probe_t         m_probe_ginput;
                probe_t         m_probe_gparam;
        };
}
