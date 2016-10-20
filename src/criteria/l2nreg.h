#pragma once

#include "average.h"
#include "softmax.h"

namespace nano
{
        ///
        /// \brief L2-norm regularized loss.
        ///
        template <typename tcriterion>
        struct l2n_criterion_t final : public tcriterion
        {
                explicit l2n_criterion_t(const string_t& configuration = string_t());

                virtual rcriterion_t clone() const override;

                virtual scalar_t value() const override;
                virtual vector_t vgrad() const override;
                virtual bool can_regularize() const override;
        };

        template <typename tcriterion>
        l2n_criterion_t<tcriterion>::l2n_criterion_t(const string_t& configuration) :
                tcriterion(configuration)
        {
        }

        template <typename tcriterion>
        rcriterion_t l2n_criterion_t<tcriterion>::clone() const
        {
                return std::make_unique<l2n_criterion_t<tcriterion>>(*this);
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

        using average_l2n_criterion_t = l2n_criterion_t<average_criterion_t>;
        using softmax_l2n_criterion_t = l2n_criterion_t<softmax_criterion_t>;
}
