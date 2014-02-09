#ifndef NANOCV_REGULARIZER_H
#define NANOCV_REGULARIZER_H

#include "util/manager.hpp"
#include "types.h"

namespace ncv
{
        class regularizer_t;

        ///
        /// \brief stores registered prototypes
        ///
        typedef manager_t<regularizer_t>                regularizer_manager_t;
        typedef regularizer_manager_t::robject_t        rregularizer_t;

        ///
        /// \brief a regulizer penalises overly-complex models,
        /// by enforcing various constraints on the \see loss function
        ///
        class regularizer_t : public clonable_t<regularizer_t>
        {
        public:

                ///
                /// \brief destructor
                ///
                virtual ~regularizer_t() {}
                
                ///
                /// \brief compute the regularization value
                /// \param x    model parameters
                /// \param gx   model gradient
                /// \return
                ///
                virtual scalar_t value(const vector_t& x, const vector_t& gx) const = 0;

                ///
                /// \brief compute the regularization value and gradient
                /// \param x    model parameters
                /// \param gx   model gradient
                /// \return
                ///
                virtual vector_t vgrad(const vector_t& x, const vector_t& gx) const = 0;
        };
}

#endif // NANOCV_REGULARIZER_H
