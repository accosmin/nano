#pragma once

#include "arch.h"
#include "tensor.h"
#include "stringi.h"
#include "util/manager.hpp"

namespace nano
{
        class layer_t;

        ///
        /// \brief stores registered prototypes
        ///
        using layer_manager_t = manager_t<layer_t>;
        using rlayer_t = layer_manager_t::trobject;
        using rlayers_t = std::vector<rlayer_t>;

        NANO_PUBLIC layer_manager_t& get_layers();

        ///
        /// \brief process a set of inputs of size (irows, icols) and produces a set of outputs of size (orows, ocols)
        ///
        class NANO_PUBLIC layer_t : public clonable_t<layer_t>
        {
        public:

                ///
                /// \brief constructor
                ///
                explicit layer_t(const string_t& parameters)
                        :       clonable_t<layer_t>(parameters)
                {
                }

                ///
                /// \brief resize to process new tensors of the given type
                ///
                virtual tensor_size_t resize(const tensor_t& tensor) = 0;

                ///
                /// \brief reset parameters to zero
                ///
                virtual void zero_params() = 0;

                ///
                /// \brief reset parameters to random values in the [min, max] range
                ///
                virtual void random_params(scalar_t min, scalar_t max) = 0;

                ///
                /// \brief serialize parameters (to memory)
                ///
                virtual scalar_t* save_params(scalar_t* params) const = 0;
                virtual const scalar_t* load_params(const scalar_t* params) = 0;

                ///
                /// \brief compute the output
                ///
                virtual const tensor_t& output(const tensor_t& input) = 0;

                ///
                /// \brief compute the gradient wrt the inputs
                ///
                virtual const tensor_t& ginput(const tensor_t& output) = 0;

                ///
                /// \brief compute the gradient wrt the parameters
                ///
                virtual void gparam(const tensor_t& output, scalar_t* gradient) = 0;

                ///
                /// \brief returns the input/output dimensions
                ///
                virtual tensor_size_t idims() const = 0;
                virtual tensor_size_t irows() const = 0;
                virtual tensor_size_t icols() const = 0;

                virtual tensor_size_t odims() const = 0;
                virtual tensor_size_t orows() const = 0;
                virtual tensor_size_t ocols() const = 0;

                ///
                /// \brief returns the number of (optimization) parameters
                ///
                virtual tensor_size_t psize() const = 0;
        };
}

