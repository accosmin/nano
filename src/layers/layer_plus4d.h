#pragma once

#include "layer.h"

namespace nano
{
        ///
        /// \brief add together multiple 4D inputs of the same size.
        ///
        class plus4d_layer_t final : public layer_t
        {
        public:

                rlayer_t clone() const final;
                json_reader_t& config(json_reader_t& reader) final { return reader; }
                json_writer_t& config(json_writer_t& writer) const final { return writer; }

                bool resize(const tensor3d_dims_t& idims) final;

                void output(tensor4d_cmap_t idata, vector_cmap_t pdata, tensor4d_map_t odata) final;
                void ginput(tensor4d_map_t idata, vector_cmap_t pdata, tensor4d_cmap_t odata) final;
                void gparam(tensor4d_cmap_t idata, vector_map_t pdata, tensor4d_cmap_t odata) final;

                tensor3d_dim_t idims() const final { return m_idims; }
                tensor3d_dim_t odims() const final { return m_odims; }

                tensor_size_t fanin() const final { return std::get<1>(idims()) / std::get<1>(odims()); }
                tensor_size_t psize() const final { return 0; }
                tensor_size_t flops_output() const final { return 10 * isize(); }
                tensor_size_t flops_ginput() const final { return 10 * isize(); }
                tensor_size_t flops_gparam() const final { return 0; }

        private:

                // attributes
                tensor3d_dim_t m_idims{{0, 0, 0}};
                tensor3d_dim_t m_odims{{0, 0, 0}};
        };
}
