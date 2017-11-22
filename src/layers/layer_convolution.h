#pragma once

#include "layer.h"
#include "conv4d.h"

namespace nano
{
        ///
        /// \brief fully-connected convolution layer as in convolution networks.
        ///
        /// parameters:
        ///     omaps   - number of output feature planes
        ///     krows   - convolution size
        ///     kcols   - convolution size
        ///     kconn   - connectivity factor: default = 1 (fully connected)
        ///     kdrow   - stride factor for the vertical axis: default = 1
        ///     kdcol   - stride factor for the horizontal axis: default = 1
        ///
        class convolution_layer_t final : public layer_t
        {
        public:

                rlayer_t clone() const final;
                json_reader_t& config(json_reader_t& reader) final;
                json_writer_t& config(json_writer_t& writer) const final;

                bool resize(const tensor3d_dims_t& idims, const string_t& name) final;
                void output(const tensor4d_cmap_t& idata, const vector_cmap_t& pdata, tensor4d_map_t&& odata) final;
                void ginput(tensor4d_map_t&& idata, const vector_cmap_t& pdata, const tensor4d_cmap_t& odata) final;
                void gparam(const tensor4d_cmap_t& idata, vector_map_t&& pdata, const tensor4d_cmap_t& odata) final;

                tensor_size_t fanin() const final;
                tensor3d_dims_t idims() const final { return m_kernel.params().idims(); }
                tensor3d_dims_t odims() const final { return m_kernel.params().odims(); }
                tensor1d_dims_t pdims() const final { return m_kernel.params().pdims(); }

                const probe_t& probe_output() const final { return m_probe_output; }
                const probe_t& probe_ginput() const final { return m_probe_ginput; }
                const probe_t& probe_gparam() const final { return m_probe_gparam; }

        private:

                auto kdims() const { return m_kernel.params().kdims(); }
                auto ksize() const { return nano::size(kdims()); }
                auto bsize() const { return m_kernel.params().bdims(); }

                template <typename tvector>
                auto kdata(tvector&& pdata) const { return map_tensor(pdata.data(), kdims()); }

                template <typename tvector>
                auto bdata(tvector&& pdata) const { return map_vector(pdata.data() + ksize(), bsize()); }

                // attributes
                conv3d_params_t m_params;
                conv4d_t        m_kernel;
                probe_t         m_probe_output;
                probe_t         m_probe_ginput;
                probe_t         m_probe_gparam;
        };
}
