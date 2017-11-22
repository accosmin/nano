#pragma once

#include "arch.h"
#include "tensor.h"
#include "factory.h"
#include "chrono/probe.h"
#include "configurable.h"

namespace nano
{
        ///
        /// \brief stores registered prototypes.
        ///
        class layer_t;
        using layer_factory_t = factory_t<layer_t>;
        using rlayer_t = layer_factory_t::trobject;

        NANO_PUBLIC layer_factory_t& get_layers();

        ///
        /// \brief computation node.
        ///
        class NANO_PUBLIC layer_t : public configurable_t
        {
        public:

                ///
                /// \brief copy the current object
                ///
                virtual rlayer_t clone() const = 0;

                ///
                /// \brief configure to process tensors of the given size
                ///
                virtual bool resize(const tensor3d_dims_t& idims, const string_t& name) = 0;
                virtual bool resize(const std::vector<tensor3d_dims_t>& idims, const string_t& name)
                {
                        return idims.size() == 1 && resize(idims[0], name);
                }

                ///
                /// \brief compute the output (given the input & the parameters)
                ///
                virtual void output(const tensor4d_cmap_t& idata, const vector_cmap_t& pdata, tensor4d_map_t&& odata) = 0;

                ///
                /// \brief compute the gradient wrt the inputs (given the output & the parameters)
                ///
                virtual void ginput(tensor4d_map_t&& idata, const vector_cmap_t& pdata, const tensor4d_cmap_t& odata) = 0;

                ///
                /// \brief compute the (cumulated) gradient wrt the parameters (given the output & the input)
                ///
                virtual void gparam(const tensor4d_cmap_t& idata, vector_map_t&& pdata, const tensor4d_cmap_t& odata) = 0;

                ///
                /// \brief number of inputs per processing unit (e.g. neuron, convolution kernel)
                ///
                virtual tensor_size_t fanin() const = 0;

                ///
                /// \brief returns the input/output/parameters dimensions
                ///
                virtual tensor3d_dims_t idims() const = 0;
                virtual tensor3d_dims_t odims() const = 0;
                virtual tensor1d_dims_t pdims() const = 0;

                ///
                /// \brief returns the timing probes for the three basic operations (output & its gradients)
                ///
                virtual const probe_t& probe_output() const = 0;
                virtual const probe_t& probe_ginput() const = 0;
                virtual const probe_t& probe_gparam() const = 0;

                ///
                /// \brief convenience functions
                ///
                tensor_size_t isize() const { return nano::size(idims()); }
                tensor_size_t osize() const { return nano::size(odims()); }
                tensor_size_t psize() const { return nano::size(pdims()); }
        };
}
