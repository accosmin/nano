#pragma once

#include "layer.h"
#include "norm4d.h"

namespace nano
{
        ///
        /// \brief normalize layer: transforms the input tensor to have zero mean and one variance,
        ///     either globally or for each feature map independently.
        ///
        class norm3d_layer_t final : public layer_t
        {
        public:

                rlayer_t clone() const final;
                json_reader_t& config(json_reader_t&) final;
                json_writer_t& config(json_writer_t&) const final;

                bool resize(const tensor3d_dims_t& idims) final;

                void output(tensor4d_cmaps_t idata, vector_cmap_t pdata, tensor4d_map_t odata) final;
                void ginput(tensor4d_maps_t idata, vector_cmap_t pdata, tensor4d_cmap_t odata) final;
                void gparam(tensor4d_cmaps_t idata, vector_map_t pdata, tensor4d_cmap_t odata) final;

                tensor_size_t fanin() const final { return 1; }
                tensor_size_t psize() const final { return 0; }
                tensor3d_dim_t odims() const final { return m_params.xdims(); }
                tensor_size_t flops_output() const final { return m_params.flops_output(); }
                tensor_size_t flops_ginput() const final { return m_params.flops_ginput(); }
                tensor_size_t flops_gparam() const final { return m_params.flops_gparam(); }

        private:

                // attributes
                norm3d_params_t m_params;
                norm4d_t        m_kernel;
        };
}
