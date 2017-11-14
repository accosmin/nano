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

                ///
                /// \brief change parameters
                ///
                void param(const tensor1d_cmap_t&);

                ///
                /// \brief compute the output
                ///
                const tensor4d_t& output(const tensor4d_t& input);

                ///
                /// \brief compute the gradient wrt the inputs
                ///
                const tensor4d_t& ginput(const tensor4d_t& output);

                ///
                /// \brief compute the (cumulated) gradient wrt the parameters
                ///
                const tensor1d_t& gparam(const tensor4d_t& output);

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
                /// \brief access functions
                ///
                const tensor4d_t& input() const { return m_idata; }
                const tensor1d_t& param() const { return m_pdata; }
                const tensor4d_t& output() const { return m_odata; }
                const tensor1d_t& gparam() const { return m_gdata; }

                tensor_size_t isize() const { return nano::size(idims()); }
                tensor_size_t osize() const { return nano::size(odims()); }
                tensor_size_t psize() const { return nano::size(pdims()); }

        protected:

                virtual void output(const tensor4d_t& idata, const tensor1d_t& pdata, tensor4d_t& odata) = 0;
                virtual void ginput(tensor4d_t& idata, const tensor1d_t& pdata, const tensor4d_t& odata) = 0;
                virtual void gparam(const tensor4d_t& idata, tensor1d_t& pdata, const tensor4d_t& odata) = 0;

        private:

                // attributes
                tensor1d_t      m_pdata;        ///< parameters buffer
                tensor1d_t      m_gdata;        ///< cumulated parameters gradient buffer
                tensor4d_t      m_idata;        ///< inputs (or its gradient) buffer
                tensor4d_t      m_odata;        ///< outputs (or its gradient) buffer
        };
}
