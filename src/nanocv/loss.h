#ifndef NANOCV_LOSS_H
#define NANOCV_LOSS_H

#include "common/manager.hpp"
#include "types.h"

namespace ncv
{
        class loss_t;

        ///
        /// \brief stores registered prototypes
        ///
        typedef manager_t<loss_t>                       loss_manager_t;
        typedef loss_manager_t::robject_t               rloss_t;

        ///
        /// \brief target value of the positive class
        ///
        inline scalar_t pos_target() { return +1.0; }

        ///
        /// \brief target value of the negative class
        ///
        inline scalar_t neg_target() { return +0.0; }
        
        ///
        /// \brief check if a target value maps to a positive class
        ///
        inline bool is_pos_target(scalar_t target) { return target > 0.5; }

        ///
        /// \brief target value for multi-class classification problems with [n_labels] classes
        ///
        vector_t class_target(size_t ilabel, size_t n_labels);

        ///
        /// \brief multivariate L1 regression error
        ///
        scalar_t l1_error(const vector_t& targets, const vector_t& scores);

        ///
        /// \brief multivariate classification error: highest score corresponds to the target class
        ///
        scalar_t mclass_error(const vector_t& targets, const vector_t& scores);

        ///
        /// \brief retrieve the predicted class indices
        ///
        indices_t classes(const vector_t& scores);

        ///
        /// \brief generic multivariate loss function of two parameters:
        /// the target value to predict (ground truth, annotation) and
        /// the current score estimation (model output).
        ///
        /// the loss function upper-bounds/approximates
        /// the true (usually non-smooth) error function to minimize.
        ///
        class loss_t : public clonable_t<loss_t>
        {
        public:

                loss_t(const string_t& configuration = string_t(),
                       const string_t& description = string_t())
                        :       clonable_t<loss_t>(configuration, description)
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
        };
}

#endif // NANOCV_LOSS_H
