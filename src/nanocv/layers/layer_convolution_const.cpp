#include "layer_convolution_const.h"

namespace ncv
{
        kconv_layer_t::kconv_layer_t(const string_t& parameters)
                :       conv_layer_t(parameters,
                                     "c(k)onstant convolution layer, parameters: dims=16[1,256],rows=8[1,32],cols=8[1,32]")
        {
        }

        scalar_t* kconv_layer_t::save_params(scalar_t* params) const
        {
                return params;
        }

        const scalar_t* kconv_layer_t::load_params(const scalar_t* params)
        {
                return params;
        }

        const tensor_t& kconv_layer_t::output(const tensor_t& input)
        {
                return conv_layer_t::output(input);
        }        

        const tensor_t& kconv_layer_t::igrad(const tensor_t& output)
        {
                return conv_layer_t::igrad(output);
        }

        void kconv_layer_t::pgrad(const tensor_t& output, scalar_t*)
        {
                assert(odims() == output.dims());
                assert(orows() == output.rows());
                assert(ocols() == output.cols());
        }
}


