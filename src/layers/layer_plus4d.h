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

                void output(tensor4d_cmaps_t idata, vector_cmap_t pdata, tensor4d_map_t odata) final;
                void ginput(tensor4d_maps_t idata, vector_cmap_t pdata, tensor4d_cmap_t odata) final;
                void gparam(tensor4d_cmaps_t idata, vector_map_t pdata, tensor4d_cmap_t odata) final;

                tensor_size_t fanin() const final { return m_fanin; }
                tensor_size_t psize() const final { return 0; }
                tensor3d_dim_t odims() const final { return m_odims; }
                tensor_size_t flops_output() const final { return 2 * m_isize; }
                tensor_size_t flops_ginput() const final { return m_isize; }
                tensor_size_t flops_gparam() const final { return 0; }

        private:

                // attributes
                tensor3d_dims_t m_idims;
                tensor_size_t   m_fanin{0};
                tensor_size_t   m_isize{0};
                tensor3d_dim_t  m_odims{{0, 0, 0}};
        };
}
