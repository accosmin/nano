#pragma once

#include "layer_convolution.h"

namespace ncv
{
        ///
        /// \brief constant convolution layer, with fixed random convolutions so no parameters to be optimized
        ///
        /// parameters:
        ///     dims=16[1,256]          - number of convolutions (output dimension)
        ///     rows=8[1,32]            - convolution size
        ///     cols=8[1,32]            - convolution size
        ///
        class kconv_layer_t : public conv_layer_t
        {
        public:

                NANOCV_MAKE_CLONABLE(kconv_layer_t)

                // constructor
                kconv_layer_t(const string_t& parameters = string_t());

                // serialize parameters
                virtual scalar_t* save_params(scalar_t* params) const;
                virtual const scalar_t* load_params(const scalar_t* params);

                // process inputs (compute outputs & gradients)
                virtual const tensor_t& output(const tensor_t& input);
                virtual const tensor_t& igrad(const tensor_t& output);
                virtual void pgrad(const tensor_t& output, scalar_t* gradient);

                // access functions
                virtual size_t psize() const { return 0; }
        };
}
