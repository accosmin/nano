#pragma once

#include "layer.h"
#include "affine4d.h"

namespace nano
{
        ///
        /// \brief fully-connected affine layer  (as in MLP models).
        ///
        /// parameters:
        ///     dims    - number of output dimensions
        ///
        struct affine_layer_t final : public layer_t
        {
                explicit affine_layer_t(const string_t& params = string_t());

                virtual rlayer_t clone() const override;
                virtual void configure(const tensor3d_dims_t&, const string_t&, tensor3d_dims_t&, tensor1d_dims_t&) override;
                virtual void output(const tensor4d_t&, const tensor1d_t&, tensor4d_t&) override;
                virtual void ginput(tensor4d_t&, const tensor1d_t&, const tensor4d_t&) override;
                virtual void gparam(const tensor4d_t&, tensor1d_t&, const tensor4d_t&) override;

                virtual tensor_size_t fanin() const override;
                virtual const probe_t& probe_output() const override { return m_probe_output; }
                virtual const probe_t& probe_ginput() const override { return m_probe_ginput; }
                virtual const probe_t& probe_gparam() const override { return m_probe_gparam; }

        private:

                tensor_size_t wsize() const { return osize() * isize(); }

                template <typename tmap>
                auto wdata(tmap param) const { return map_matrix(param.data(), osize(), isize()); }
                template <typename tmap>
                auto bdata(tmap param) const { return map_vector(param.data() + wsize(), osize()); }

                // attributes
                probe_t         m_probe_output;
                probe_t         m_probe_ginput;
                probe_t         m_probe_gparam;
        };
}
