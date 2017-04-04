#include "activation.h"
#include "math/numeric.h"
#include "text/to_params.h"
#include "text/from_params.h"

namespace nano
{
        activation_layer_t::activation_layer_t(const string_t& parameters) :
                layer_t(parameters),
                m_idims({0, 0, 0}),
                m_odims({0, 0, 0})
        {
        }

        void activation_layer_t::configure(const tensor3d_dims_t& idims)
        {
                m_idims = idims;
                m_odims = idims;
        }

        void activation_layer_t::output(tensor3d_const_map_t idata, tensor1d_const_map_t param, tensor3d_map_t odata)
        {
                assert(idata.dims() == idims());
                assert(param.size() == psize());
                assert(odata.dims() == odims());
                NANO_UNUSED1_RELEASE(param);

                aoutput(idata.array(), odata.array());
        }

        void activation_layer_t::ginput(tensor3d_map_t idata, tensor1d_const_map_t param, tensor3d_const_map_t odata)
        {
                assert(idata.dims() == idims());
                assert(param.size() == psize());
                assert(odata.dims() == odims());
                NANO_UNUSED1_RELEASE(param);

                aginput(idata.array(), odata.array());
        }

        void activation_layer_t::gparam(tensor3d_const_map_t idata, tensor1d_map_t param, tensor3d_const_map_t odata)
        {
                assert(idata.dims() == idims());
                assert(param.size() == psize());
                assert(odata.dims() == odims());
                NANO_UNUSED3_RELEASE(idata, param, odata);
        }

        rlayer_t activation_layer_sine_t::clone() const
        {
                return std::make_unique<activation_layer_sine_t>(*this);
        }

        void activation_layer_sine_t::aoutput(tensor3d_const_array_t idata, tensor3d_array_t odata) const
        {
                odata = idata.sin();
        }

        void activation_layer_sine_t::aginput(tensor3d_array_t idata, tensor3d_const_array_t odata) const
        {
                idata = odata * idata.cos();
        }

        rlayer_t activation_layer_snorm_t::clone() const
        {
                return std::make_unique<activation_layer_snorm_t>(*this);
        }

        void activation_layer_snorm_t::aoutput(tensor3d_const_array_t idata, tensor3d_array_t odata) const
        {
                odata = idata / (1 + idata.square()).sqrt();
        }

        void activation_layer_snorm_t::aginput(tensor3d_array_t idata, tensor3d_const_array_t odata) const
        {
                idata = odata / (1 + idata.square()).cube().sqrt();
        }

        rlayer_t activation_layer_tanh_t::clone() const
        {
                return std::make_unique<activation_layer_tanh_t>(*this);
        }

        void activation_layer_tanh_t::aoutput(tensor3d_const_array_t idata, tensor3d_array_t odata) const
        {
                odata = idata.tanh();
        }

        void activation_layer_tanh_t::aginput(tensor3d_array_t idata, tensor3d_const_array_t odata) const
        {
                idata = odata * 4 / (idata.exp() + (-idata).exp()).square();
        }

        rlayer_t activation_layer_sigm_t::clone() const
        {
                return std::make_unique<activation_layer_sigm_t>(*this);
        }

        void activation_layer_sigm_t::aoutput(tensor3d_const_array_t idata, tensor3d_array_t odata) const
        {
                odata = idata.exp() / (1 + idata.exp());
        }

        void activation_layer_sigm_t::aginput(tensor3d_array_t idata, tensor3d_const_array_t odata) const
        {
                idata = odata * idata.exp() / (1 + idata.exp()).square();
        }

        rlayer_t activation_layer_splus_t::clone() const
        {
                return std::make_unique<activation_layer_splus_t>(*this);
        }

        void activation_layer_splus_t::aoutput(tensor3d_const_array_t idata, tensor3d_array_t odata) const
        {
                odata = (1 + idata.exp()).log();
        }

        void activation_layer_splus_t::aginput(tensor3d_array_t idata, tensor3d_const_array_t odata) const
        {
                idata = odata * idata.exp() / (1 + idata.exp());
        }

        rlayer_t activation_layer_unit_t::clone() const
        {
                return std::make_unique<activation_layer_unit_t>(*this);
        }

        void activation_layer_unit_t::aoutput(tensor3d_const_array_t idata, tensor3d_array_t odata) const
        {
                odata = idata;
        }

        void activation_layer_unit_t::aginput(tensor3d_array_t idata, tensor3d_const_array_t odata) const
        {
                idata = odata;
        }

        rlayer_t activation_layer_pwave_t::clone() const
        {
                return std::make_unique<activation_layer_pwave_t>(*this);
        }

        void activation_layer_pwave_t::aoutput(tensor3d_const_array_t idata, tensor3d_array_t odata) const
        {
                odata = idata / (1 + idata.square());
        }

        void activation_layer_pwave_t::aginput(tensor3d_array_t idata, tensor3d_const_array_t odata) const
        {
                idata = odata * (1 - idata.square()) / (1 + idata.square()).square();
        }
}
