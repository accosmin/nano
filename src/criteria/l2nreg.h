#pragma once

#include "average.h"
#include "softmax.h"

namespace nano
{
        ///
        /// \brief L2-norm regularized loss.
        ///
        template <typename tcriterion>
        class l2n_criterion_t : public tcriterion
        {
        public:

                ///
                /// \brief constructor
                ///
                explicit l2n_criterion_t(const string_t& configuration = string_t());

                ///
                /// \brief cumulated loss value
                ///
                virtual scalar_t value() const override;

                ///
                /// \brief cumulated gradient
                ///
                virtual vector_t vgrad() const override;

                ///
                /// \brief check if the criterion has a regularization term to tune
                ///
                virtual bool can_regularize() const override;
        };

        template <typename tcriterion>
        l2n_criterion_t<tcriterion>::l2n_criterion_t(const string_t& configuration) :
                tcriterion(configuration)
        {
        }

        template <typename tcriterion>
        scalar_t l2n_criterion_t<tcriterion>::value() const
        {
                return  this->lweight() * (tcriterion::value()) +
                        this->rweight() * (scalar_t(0.5) * this->params().squaredNorm() / scalar_t(this->psize()));
        }

        template <typename tcriterion>
        vector_t l2n_criterion_t<tcriterion>::vgrad() const
        {
                return  this->lweight() * (tcriterion::vgrad()) +
                        this->rweight() * (this->params() / scalar_t(this->psize()));
        }

        template <typename tcriterion>
        bool l2n_criterion_t<tcriterion>::can_regularize() const
        {
                return true;
        }

        struct average_l2n_criterion_t : public l2n_criterion_t<average_criterion_t>
        {
                NANO_MAKE_CLONABLE(average_l2n_criterion_t)

                explicit average_l2n_criterion_t(const string_t& configuration = string_t()) :
                        l2n_criterion_t<average_criterion_t>(configuration)
                {
                }
        };

        struct softmax_l2n_criterion_t : public l2n_criterion_t<softmax_criterion_t>
        {
                NANO_MAKE_CLONABLE(softmax_l2n_criterion_t)

                explicit softmax_l2n_criterion_t(const string_t& configuration = string_t()) :
                        l2n_criterion_t<softmax_criterion_t>(configuration)
                {
                }
        };
}
