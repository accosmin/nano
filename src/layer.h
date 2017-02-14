#pragma once

#include <iosfwd>
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
                /// \brief resize to process new tensors of the given type
                ///
                virtual tensor_size_t resize(const tensor3d_t& tensor) = 0;

                ///
                /// \brief reset parameters to zero
                ///
                virtual void zero_params() = 0;

                ///
                /// \brief reset parameters to random values in the [min, max] range
                ///
                virtual void random_params(const scalar_t min, const scalar_t max) = 0;

                ///
                /// \brief serialize parameters (to memory)
                ///
                virtual scalar_t* save_params(scalar_t* params) const = 0;
                virtual const scalar_t* load_params(const scalar_t* params) = 0;

                ///
                /// \brief serialize to disk
                ///
                virtual bool save(std::ostream&) const = 0;
                virtual bool load(std::istream&) = 0;

                ///
                /// \brief compute the output
                ///
                virtual const tensor3d_t& output(const tensor3d_t& input) = 0;

                ///
                /// \brief compute the gradient wrt the inputs
                ///
                virtual const tensor3d_t& ginput(const tensor3d_t& output) = 0;

                ///
                /// \brief compute the gradient wrt the parameters
                ///
                virtual void gparam(const tensor3d_t& output, scalar_t* gradient) = 0;

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

