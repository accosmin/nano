#pragma once

#include "layer.h"
#include "affine4d.h"

namespace nano
{
        ///
        /// \brief fully-connected affine layer  (as in MLP models).
        ///
        /// parameters:
        ///     omaps   - number of output feature maps
        ///     orows   - number of output rows (=1)
        ///     ocols   - number of output cols (=1)
        ///
        struct affine_layer_t final : public layer_t
        {
                explicit affine_layer_t(const string_t& params = string_t());

                virtual rlayer_t clone() const override;
                virtual void configure(const tensor3d_dims_t& idims, const string_t& name) override;
                virtual void output(const tensor4d_t& idata, const tensor1d_t& pdata, tensor4d_t& odata) override;
                virtual void ginput(tensor4d_t& idata, const tensor1d_t& pdata, const tensor4d_t& odata) override;
                virtual void gparam(const tensor4d_t& idata, tensor1d_t& pdata, const tensor4d_t& odata) override;

                virtual tensor_size_t fanin() const override;
                virtual tensor3d_dims_t idims() const override { return m_kernel.params().idims(); }
                virtual tensor3d_dims_t odims() const override { return m_kernel.params().odims(); }
                virtual tensor1d_dims_t pdims() const override { return m_kernel.params().pdims(); }

                virtual const probe_t& probe_output() const override { return m_probe_output; }
                virtual const probe_t& probe_ginput() const override { return m_probe_ginput; }
                virtual const probe_t& probe_gparam() const override { return m_probe_gparam; }

        private:

                auto wsize() const { return osize() * isize(); }

                auto wdata(tensor1d_t& pdata) const { return map_matrix(pdata.data(), osize(), isize()); }
                auto bdata(tensor1d_t& pdata) const { return map_vector(pdata.data() + wsize(), osize()); }

                auto wdata(const tensor1d_t& pdata) const { return map_matrix(pdata.data(), osize(), isize()); }
                auto bdata(const tensor1d_t& pdata) const { return map_vector(pdata.data() + wsize(), osize()); }

                // attributes
                affine4d_t      m_kernel;
                probe_t         m_probe_output;
                probe_t         m_probe_ginput;
                probe_t         m_probe_gparam;
        };
}
