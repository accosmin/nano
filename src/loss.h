#pragma once

#include "arch.h"
#include "tensor.h"
#include "factory.h"

namespace nano
{
        ///
        /// \brief stores registered prototypes
        ///
        struct loss_t;
        using loss_factory_t = factory_t<loss_t>;
        using rloss_t = loss_factory_t::trobject;

        NANO_PUBLIC loss_factory_t& get_losses();

        ///
        /// \brief generic multivariate loss function of two parameters:
        /// the target value to predict (ground truth, annotation) and
        /// the current score estimation (model output).
        ///
        /// the loss function upper-bounds/approximates
        /// the true (usually non-smooth) error function to minimize.
        ///
        struct NANO_PUBLIC loss_t : public configurable_t
        {
                using configurable_t::configurable_t;

                ///
                /// \brief compute the error value
                ///
                virtual scalar_t error(const vector_t& targets, const vector_t& scores) const = 0;

                ///
                /// \brief compute the loss value (an upper bound of the usually non-continuous error function)
                ///
                virtual scalar_t value(const vector_t& targets, const vector_t& scores) const = 0;

                ///
                /// \brief compute the loss gradient (wrt the scores)
                ///
                virtual vector_t vgrad(const vector_t& targets, const vector_t& scores) const = 0;

                ///
                /// \brief predicted label indices (if a classification problem)
                ///
                virtual indices_t labels(const vector_t& scores) const = 0;
        };
}
