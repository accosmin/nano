#ifndef NANOCV_LOSS_H
#define NANOCV_LOSS_H

#include "util/manager.hpp"
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
        /// \return
        ///
        inline scalar_t pos_target() { return +1.0; }

        ///
        /// \brief target value of the negative class
        /// \return
        ///
        inline scalar_t neg_target() { return -1.0; }

        ///
        /// \brief target value for multi-class classification problems
        /// \param ilabel
        /// \param n_labels
        /// \return
        ///
        vector_t class_target(size_t ilabel, size_t n_labels);

        ///
        /// \brief multivariate L1 regression error
        /// \param targets
        /// \param scores
        /// \return
        ///
        scalar_t l1_error(const vector_t& targets, const vector_t& scores);

        ///
        /// \brief multivariate classification error: matches the sign of predictions vs target classes
        /// \param targets
        /// \param scores
        /// \return
        ///
        scalar_t eclass_error(const vector_t& targets, const vector_t& scores);

        ///
        /// \brief multivariate classification error: highest score corresponds to the target class
        /// \param targets
        /// \param scores
        /// \return
        ///
        scalar_t mclass_error(const vector_t& targets, const vector_t& scores);

        ///
        /// \brief generic multivariate loss function of two parameters:
        /// the target value to predict and the current score estimation.
        ///
        /// the loss function upper-bounds/approximates
        /// the true (usually non-smooth) error function to minimize.
        ///
        class loss_t : public clonable_t<loss_t>
        {
        public:

                ///
                /// \brief destructur
                ///
                virtual ~loss_t() {}
                
                ///
                /// \brief compute the error value
                /// \param targets
                /// \param scores
                /// \return
                ///
                virtual scalar_t error(const vector_t& targets, const vector_t& scores) const = 0;
                
                ///
                /// \brief compute the loss value
                /// \param targets
                /// \param scores
                /// \return
                ///
                virtual scalar_t value(const vector_t& targets, const vector_t& scores) const = 0;

                ///
                /// \brief compute the loss gradient
                /// \param targets
                /// \param scores
                /// \return
                ///
                virtual vector_t vgrad(const vector_t& targets, const vector_t& scores) const = 0;
        };
}

#endif // NANOCV_LOSS_H
