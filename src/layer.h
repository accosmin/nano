#pragma once

#include "arch.h"
#include "tensor.h"
#include "factory.h"
#include "chrono/probe.h"

namespace nano
{
        ///
        /// \brief stores registered prototypes
        ///
        struct layer_t;
        using layer_factory_t = factory_t<layer_t>;
        using rlayer_t = layer_factory_t::trobject;

        NANO_PUBLIC layer_factory_t& get_layers();

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
                virtual void configure(const tensor3d_dims_t& idims, const string_t& name) = 0;

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
                /// \brief returns the timing probes for the three basic operations (output & its gradients)
                ///
                virtual const probe_t& probe_output() const = 0;
                virtual const probe_t& probe_ginput() const = 0;
                virtual const probe_t& probe_gparam() const = 0;

                ///
                /// \brief convenience overloads for the basic operations (output & its gradients)
                ///
                void output(const scalar_t* idata, const scalar_t* param, scalar_t* odata);
                void ginput(scalar_t* idata, const scalar_t* param, const scalar_t* odata);
                void gparam(const scalar_t* idata, scalar_t* param, const scalar_t* odata);

                ///
                /// \brief returns the input/output size
                ///
                auto isize() const { return nano::size(idims()); }
                auto osize() const { return nano::size(odims()); }
                auto xsize() const { return isize() + osize(); }
        };

        inline void layer_t::output(const scalar_t* idata, const scalar_t* param, scalar_t* odata)
        {
                output(map_tensor(idata, idims()), map_tensor(param, psize()), map_tensor(odata, odims()));
        }

        inline void layer_t::ginput(scalar_t* idata, const scalar_t* param, const scalar_t* odata)
        {
                ginput(map_tensor(idata, idims()), map_tensor(param, psize()), map_tensor(odata, odims()));
        }

        inline void layer_t::gparam(const scalar_t* idata, scalar_t* param, const scalar_t* odata)
        {
                gparam(map_tensor(idata, idims()), map_tensor(param, psize()), map_tensor(odata, odims()));
        }
}
