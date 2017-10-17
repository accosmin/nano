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
        class layer_t;
        using layer_factory_t = factory_t<layer_t>;
        using rlayer_t = layer_factory_t::trobject;

        NANO_PUBLIC layer_factory_t& get_layers();

        ///
        /// \brief process a set of inputs of size (irows, icols) and produces a set of outputs of size (orows, ocols).
        ///
        class NANO_PUBLIC layer_t : public configurable_t
        {
        public:
                ///
                /// \brief constructor
                ///
                layer_t(const string_t& config = string_t());

                ///
                /// \brief create a copy of the current object
                ///
                virtual rlayer_t clone() const = 0;

                ///
                /// \brief configure to process new tensors of the given size
                ///
                void configure(const tensor3d_dims_t& idims, const string_t& name);

                ///
                /// \brief change parameters
                ///
                void param(const tensor1d_const_map_t&);

                ///
                /// \brief compute the output
                ///
                const tensor4d_t& output(const tensor4d_t& idata);

                ///
                /// \brief compute the gradient wrt the inputs
                ///
                const tensor4d_t& ginput(const tensor4d_t& odata);

                ///
                /// \brief compute the (cumulated) gradient wrt the parameters
                ///
                const tensor1d_t& gparam(const tensor4d_t& odata);

                ///
                /// \brief number of inputs per processing unit (e.g. neuron, convolution kernel)
                ///
                virtual tensor_size_t fanin() const = 0;

                ///
                /// \brief returns the input/output dimensions
                ///
                tensor3d_dims_t idims() const { return m_idims; }
                tensor3d_dims_t odims() const { return m_odims; }

                ///
                /// \brief returns the number of parameters to optimize
                ///
                tensor1d_dims_t pdims() const { return m_pdims; }
                tensor_size_t psize() const { return nano::size(m_pdims; }

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
                const tensor1d_t& param() const { return m_param; }
                const tensor4d_t& output() const { return m_odata; }
                const tensor1d_t& gparam() const { return m_gparam; }

        protected:

                virtual void configure(const tensor3d_dims_t& idims, const string_t& name,
                        tensor3d_dims_t& odims, tensor1d_dims_t& pdims) = 0;

                virtual void output(const tensor4d_t& idata, const tensor1d_t& param, tensor4d_t& odata) const = 0;
                virtual void ginput(tensor4d_t& idata, const tensor1d_t& param, const tensor4d_t& odata) const = 0;
                virtual void gparam(const tensor4d_t& idata, tensor1d_t& param, const tensor4d_t& odata) const = 0;

        private:

                // attributes
                tensor3d_dims_t m_idims;        ///<
                tensor1d_dims_t m_pdims;        ///<
                tensor3d_dims_t m_odims;        ///<

                tensor1d_t      m_param;        ///< parameters buffer
                tensor1d_t      m_gparam;       ///< cumulated parameters gradient buffer
                tensor4d_t      m_idata;        ///< inputs (or its gradient) buffer
                tensor4d_t      m_odata;        ///< outputs (or its gradient) buffer
        };
}
