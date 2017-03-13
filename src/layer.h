#pragma once

#include "arch.h"
#include "tensor.h"
#include "manager.h"

namespace nano
{
        ///
        /// \brief stores registered prototypes
        ///
        class layer_t;
        using layer_manager_t = manager_t<layer_t>;
        using rlayer_t = layer_manager_t::trobject;

        NANO_PUBLIC layer_manager_t& get_layers();

        ///
        /// \brief process a set of inputs of size (irows, icols) and produces a set of outputs of size (orows, ocols)
        ///
        class NANO_PUBLIC layer_t : public clonable_t
        {
        public:

                ///
                /// \brief constructor
                ///
                explicit layer_t(const string_t& parameters) : clonable_t(parameters) {}

                ///
                /// \brief create a copy of the current object
                ///
                virtual rlayer_t clone() const = 0;

                ///
                /// \brief configure to process new tensors of the given size
                /// \return the number of parameters
                ///
                virtual tensor_size_t configure(const dim3d_t& idims, dim3d_t& odims) = 0;

                ///
                /// \brief configure to use the given input-output-parameters buffers
                ///
                virtual bool configure(tensor3d_t& idata, tensor3d_t& odata, vector_map_t& params) = 0;

                ///
                /// \brief reset parameters to random values in the [min, max] range
                ///
                virtual void random(const scalar_t min, const scalar_t max) = 0;

                ///
                /// \brief compute the output
                ///
                virtual void output() = 0;

                ///
                /// \brief compute the gradient wrt the inputs
                ///
                virtual void ginput() = 0;

                ///
                /// \brief compute the gradient wrt the parameters
                ///
                virtual void gparam() = 0;

                ///
                /// \brief returns the input/output dimensions
                ///
                virtual dim3d_t idims() const = 0;
                virtual dim3d_t odims() const = 0;

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

