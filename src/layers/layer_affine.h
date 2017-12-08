#pragma once

#include "layer.h"
#include "affine4d.h"

namespace nano
{
        ///
        /// \brief fully-connected affine layer (as in MLP models).
        ///
        /// parameters:
        ///     omaps   - number of output feature maps
        ///     orows   - number of output rows (=1)
        ///     ocols   - number of output cols (=1)
        ///
        class affine_layer_t final : public layer_t
        {
        public:

                rlayer_t clone() const final;
                json_reader_t& config(json_reader_t&) final;
                json_writer_t& config(json_writer_t&) const final;

                bool resize(const tensor3d_dims_t& idims) final;

                void output(tensor4d_cmap_t idata, vector_cmap_t pdata, tensor4d_map_t odata) final;
                void ginput(tensor4d_map_t idata, vector_cmap_t pdata, tensor4d_cmap_t odata) final;
                void gparam(tensor4d_cmap_t idata, vector_map_t pdata, tensor4d_cmap_t odata) final;

                tensor3d_dim_t idims() const final { return m_params.idims(); }
                tensor3d_dim_t odims() const final { return m_params.odims(); }

                tensor_size_t fanin() const final { return m_params.isize(); }
                tensor_size_t psize() const final { return m_params.psize(); }
                tensor_size_t flops_output() const final { return m_params.flops_output(); }
                tensor_size_t flops_ginput() const final { return m_params.flops_ginput(); }
                tensor_size_t flops_gparam() const final { return m_params.flops_gparam(); }

        private:

                auto wsize() const { return osize() * isize(); }

                template <typename tvector>
                auto wdata(tvector&& pdata) const { return map_matrix(pdata.data(), osize(), isize()); }

                template <typename tvector>
                auto bdata(tvector&& pdata) const { return map_vector(pdata.data() + wsize(), osize()); }

                // attributes
                affine_params_t m_params;
                affine4d_t      m_kernel;
        };
}
