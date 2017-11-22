#pragma once

#include "layer.h"
#include "norm4d.h"

namespace nano
{
        ///
        /// \brief normalize layer: transforms the input tensor to have zero mean and one variance,
        ///     either globally or for each feature map.
        ///
        class normalize_layer_t final : public layer_t
        {
        public:

                using layer_t::resize;

                rlayer_t clone() const final;
                json_reader_t& config(json_reader_t&) final;
                json_writer_t& config(json_writer_t&) const final;

                bool resize(const tensor3d_dims_t& idims, const string_t& name) final;
                void output(const tensor4d_cmap_t& idata, const vector_cmap_t& pdata, tensor4d_map_t&& odata) final;
                void ginput(tensor4d_map_t&& idata, const vector_cmap_t& pdata, const tensor4d_cmap_t& odata) final;
                void gparam(const tensor4d_cmap_t& idata, vector_map_t&& pdata, const tensor4d_cmap_t& odata) final;

                tensor_size_t fanin() const final { return 1; }
                tensor3d_dims_t idims() const final { return m_params.xdims(); }
                tensor3d_dims_t odims() const final { return m_params.xdims(); }
                tensor1d_dims_t pdims() const final { return {0}; }

                const probe_t& probe_output() const final { return m_probe_output; }
                const probe_t& probe_ginput() const final { return m_probe_ginput; }
                const probe_t& probe_gparam() const final { return m_probe_gparam; }

        private:

                // attributes
                norm3d_params_t m_params;
                norm4d_t        m_kernel;
                probe_t         m_probe_output;
                probe_t         m_probe_ginput;
                probe_t         m_probe_gparam;
        };
}
