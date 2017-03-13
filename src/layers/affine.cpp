#include "affine.h"
#include "text/to_params.h"
#include "text/from_params.h"

namespace nano
{
        affine_layer_t::affine_layer_t(const string_t& parameters) :
                layer_t(to_params(parameters, "dims", "10[1,4096]")),
                m_idims({0, 0, 0}),
                m_odims({0, 0, 0}),
                m_psize(0)
        {
        }

        rlayer_t affine_layer_t::clone() const
        {
                return std::make_unique<affine_layer_t>(*this);
        }

        bool affine_layer_t::configure(const dim3d_t& idims)
        {
                m_idims = idims;
                m_odims = {nano::clamp(nano::from_params<tensor_size_t>(config(), "dims"), 1, 4096), 1, 1};
                m_psize = nano::size(m_odims) * (nano::size(m_idims) + 1);
                return true;
        }

        void affine_layer_t::output(const tensor3d_map_t& idata, const tensor1d_map_t& param, const tensor3d_map_t& odata)
        {
                assert(idata.dims() == idims());
                assert(param.size() == psize());
                assert(odata.dims() == odims());

                odata.vector() = wdata(param) * idata.vector() + bdata(param);
        }

        void affine_layer_t::ginput(const tensor3d_map_t& idata, const tensor1d_map_t& param, const tensor3d_map_t& odata)
        {
                assert(idata.dims() == idims());
                assert(param.size() == psize());
                assert(odata.dims() == odims());

                idata.vector() = wdata(param).transpose() * odata.vector();
        }

        void affine_layer_t::gparam(const tensor3d_map_t& idata, const tensor1d_map_t& param, const tensor3d_map_t& odata)
        {
                assert(idata.dims() == idims());
                assert(param.size() == psize());
                assert(odata.dims() == odims());

                bdata(param) = odata.vector();
                wdata(param).noalias() = odata.vector() * idata.vector().transpose();
        }
}

