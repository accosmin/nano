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
        class normalize_layer_t final : public layer_t
        {
        public:

                rlayer_t clone() const final;
                json_reader_t& config(json_reader_t&) final;
                json_writer_t& config(json_writer_t&) const final;

                bool resize(const tensor3d_dims_t& idims, const string_t& name) final;
                void output(const tensor4d_t& idata, const tensor1d_t& pdata, tensor4d_t& odata) final;
                void ginput(tensor4d_t& idata, const tensor1d_t& pdata, const tensor4d_t& odata) final;
                void gparam(const tensor4d_t& idata, tensor1d_t& pdata, const tensor4d_t& odata) final;

                tensor_size_t fanin() const final { return 1; }
                tensor3d_dims_t idims() const final { return m_xdims; }
                tensor3d_dims_t odims() const final { return m_xdims; }
                tensor1d_dims_t pdims() const final { return {0}; }

                const probe_t& probe_output() const final { return m_probe_output; }
                const probe_t& probe_ginput() const final { return m_probe_ginput; }
                const probe_t& probe_gparam() const final { return m_probe_gparam; }

        private:

                // attributes
                norm_type       m_type;
                tensor3d_dims_t m_xdims;        ///< input/output dimensions
                probe_t         m_probe_output;
                probe_t         m_probe_ginput;
                probe_t         m_probe_gparam;
        };
}
