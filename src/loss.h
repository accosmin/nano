#pragma once

#include "arch.h"
#include "tensor.h"
#include "core/json.h"
#include "core/factory.h"

namespace nano
{
        class loss_t;
        using loss_factory_t = factory_t<loss_t>;
        using rloss_t = loss_factory_t::trobject;

        NANO_PUBLIC loss_factory_t& get_losses();

        ///
        /// \brief generic multivariate loss function of two parameters:
        ///     - the target value to predict (ground truth, annotation) and
        ///     - the model's output (prediction).
        ///
        /// the loss function upper-bounds/approximates
        /// the true (usually non-smooth) error function to minimize.
        ///
        class NANO_PUBLIC loss_t : public json_configurable_t
        {
        public:

                ///
                /// \brief serialization to JSON not needed
                ///
                void to_json(json_t&) const final {}
                void from_json(const json_t&) final {}

                ///
                /// \brief compute the error value
                ///
                virtual scalar_t error(const tensor3d_cmap_t& target, const tensor3d_cmap_t& output) const = 0;

                ///
                /// \brief compute the loss value (an upper bound of the usually non-continuous error function)
                ///
                virtual scalar_t value(const tensor3d_cmap_t& target, const tensor3d_cmap_t& output) const = 0;

                ///
                /// \brief compute the loss gradient (wrt the output)
                ///
                virtual tensor3d_t vgrad(const tensor3d_cmap_t& target, const tensor3d_cmap_t& output) const = 0;
        };
}
