#ifndef NANOCV_LAYER_H
#define NANOCV_LAYER_H

#include "common/manager.hpp"
#include "types.h"

namespace ncv
{
        class layer_t;

        ///
        /// \brief stores registered prototypes
        ///
        typedef manager_t<layer_t>                      layer_manager_t;
        typedef layer_manager_t::robject_t              rlayer_t;
        typedef std::vector<rlayer_t>                   rlayers_t;

        ///
        /// \brief process a set of inputs of size (irows, icols) and produces a set of outputs of size (orows, ocols)
        ///
        class layer_t : public clonable_t<layer_t>
        {
        public:

                ///
                /// \brief constructor
                ///
                layer_t(const string_t& parameters, const string_t& description)
                        :       clonable_t<layer_t>(parameters, description)
                {
                }

                ///
                /// \brief destructor
                ///
                virtual ~layer_t() {}

                ///
                /// \brief resize to process new tensors of the given type
                ///
                virtual size_t resize(const tensor_t& tensor) = 0;

                ///
                /// \brief reset parameters to zero
                ///
                virtual void zero_params() = 0;

                ///
                /// \brief reset parameters to random values in the [min, max] range
                ///
                virtual void random_params(scalar_t min, scalar_t max) = 0;

                ///
                /// \brief serialize parameters & gradients
                ///
                virtual ovectorizer_t& save_params(ovectorizer_t& s) const = 0;
                virtual ivectorizer_t& load_params(ivectorizer_t& s) = 0;
                virtual ovectorizer_t& save_grad(ovectorizer_t& s) const = 0;

                ///
                /// \brief compute the output tensor
                ///
                virtual const tensor_t& forward(const tensor_t& input) = 0;

                ///
                /// \brief compute the gradient tensor
                ///
                virtual const tensor_t& backward(const tensor_t& gradient) = 0;

                ///
                /// \brief returns the input/output tensor
                ///
                virtual const tensor_t& input() const = 0;
                virtual const tensor_t& output() const = 0;
        };
}

#endif // NANOCV_LAYER_H
