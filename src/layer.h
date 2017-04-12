#pragma once

#include "arch.h"
#include "tensor.h"
#include "manager.h"

namespace nano
{
        ///
        /// \brief stores registered prototypes
        ///
        struct layer_t;
        using layer_manager_t = manager_t<layer_t>;
        using rlayer_t = layer_manager_t::trobject;

        NANO_PUBLIC layer_manager_t& get_layers();

        ///
        /// \brief process a set of inputs of size (irows, icols) and produces a set of outputs of size (orows, ocols).
        ///
        struct NANO_PUBLIC layer_t : public configurable_t
        {
                using configurable_t::configurable_t;

                ///
                /// \brief create a copy of the current object
                ///
                virtual rlayer_t clone() const = 0;

                ///
                /// \brief configure to process new tensors of the given size
                ///
                virtual void configure(const tensor3d_dims_t& idims) = 0;

                ///
                /// \brief compute the output: (input, parameters, output)
                ///
                virtual void output(tensor3d_const_map_t idata, tensor1d_const_map_t param, tensor3d_map_t odata) = 0;

                ///
                /// \brief compute the gradient wrt the inputs: (input, parameters, output)
                ///
                virtual void ginput(tensor3d_map_t idata, tensor1d_const_map_t param, tensor3d_const_map_t odata) = 0;

                ///
                /// \brief compute the gradient wrt the parameters: (input, parameters, output)
                ///
                virtual void gparam(tensor3d_const_map_t idata, tensor1d_map_t param, tensor3d_const_map_t odata) = 0;

                ///
                /// \brief returns the input/output dimensions
                ///
                virtual tensor3d_dims_t idims() const = 0;
                virtual tensor3d_dims_t odims() const = 0;

                ///
                /// \brief number of inputs per processing unit (e.g. neuron, convolution kernel)
                ///
                virtual tensor_size_t fanin() const = 0;

                ///
                /// \brief returns the number of (optimization) parameters
                ///
                virtual tensor_size_t psize() const = 0;

                ///
                /// \brief returns the (approximated) FLOPs necessary to compute the output
                ///
                virtual tensor_size_t flops() const = 0;
        };
}

