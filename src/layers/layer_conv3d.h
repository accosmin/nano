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
        class conv3d_layer_t final : public layer_t
        {
        public:

                rlayer_t clone() const final;
                json_reader_t& config(json_reader_t& reader) final;
                json_writer_t& config(json_writer_t& writer) const final;

                bool resize(const tensor3d_dims_t& idims) final;

                void random(vector_map_t pdata) const final;
                void output(tensor4d_cmaps_t idata, vector_cmap_t pdata, tensor4d_map_t odata) final;
                void ginput(tensor4d_maps_t idata, vector_cmap_t pdata, tensor4d_cmap_t odata) final;
                void gparam(tensor4d_cmaps_t idata, vector_map_t pdata, tensor4d_cmap_t odata) final;

                tensor_size_t psize() const final { return m_params.psize(); }
                tensor3d_dim_t odims() const final { return m_params.odims(); }
                tensor_size_t flops_output() const final { return m_params.flops_output(); }
                tensor_size_t flops_ginput() const final { return m_params.flops_ginput(); }
                tensor_size_t flops_gparam() const final { return m_params.flops_gparam(); }

        private:

                auto kdims() const { return m_params.kdims(); }
                auto ksize() const { return m_params.ksize(); }
                auto bsize() const { return m_params.bdims(); }

                tensor_size_t imaps() const { return m_params.imaps(); }
                tensor_size_t kconn() const { return m_params.kconn(); }

                template <typename tvector>
                auto kdata(tvector&& pdata) const { return map_tensor(pdata.data(), kdims()); }

                template <typename tvector>
                auto bdata(tvector&& pdata) const { return map_vector(pdata.data() + ksize(), bsize()); }

                // attributes
                conv3d_params_t m_params;
                conv4d_t        m_kernel;
        };
}
