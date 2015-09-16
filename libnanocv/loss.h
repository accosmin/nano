#pragma once

#include "libcore/arch.h"
#include "libcore/tensor.h"
#include "libcore/manager.hpp"

namespace ncv
{
        class loss_t;

        ///
        /// \brief stores registered prototypes
        ///
        typedef manager_t<loss_t>               loss_manager_t;
        typedef loss_manager_t::robject_t       rloss_t;

        NANOCV_PUBLIC loss_manager_t& get_losses();

        ///
        /// \brief generic multivariate loss function of two parameters:
        /// the target value to predict (ground truth, annotation) and
        /// the current score estimation (model output).
        ///
        /// the loss function upper-bounds/approximates
        /// the true (usually non-smooth) error function to minimize.
        ///
        class NANOCV_PUBLIC loss_t : public clonable_t<loss_t>
        {
        public:

                explicit loss_t(const string_t& configuration = string_t())
                        :       clonable_t<loss_t>(configuration)
                {
                }

                ///
                /// \brief destructor
                ///
                virtual ~loss_t() {}
                
                ///
                /// \brief compute the error value
                ///
                virtual scalar_t error(const vector_t& targets, const vector_t& scores) const = 0;
                
                ///
                /// \brief compute the loss value (an upper bound of the usually non-continuous error function)
                ///
                virtual scalar_t value(const vector_t& targets, const vector_t& scores) const = 0;

                ///
                /// \brief compute the loss gradient
                ///
                virtual vector_t vgrad(const vector_t& targets, const vector_t& scores) const = 0;

                ///
                /// \brief predicted label indices (if classification problem)
                ///
                virtual indices_t labels(const vector_t& scores) const = 0;
        };
}
