#pragma once

#include "loss.h"
#include <cassert>

namespace nano
{
        ///
        /// \brief (multivariate) regression loss that upper-bounds the L1-distance between target and score/output.
        ///
        template
        <
                typename tvalue_op,     ///< loss value: tvalue_op(targets, scores)
                typename tvgrad_op      ///< loss gradient wrt scores: tvgrad_op(targets, scores)
        >
        class regression_loss_t : public loss_t
        {
        public:
                using tregression = regression_loss_t<tvalue_op, tvgrad_op>;

                // constructor
                explicit regression_loss_t(const string_t& parameters = string_t()) : loss_t(parameters) {}

                // destructor
                virtual ~regression_loss_t() {}

                // clone
                virtual rloss_t clone(const string_t& parameters) const override final
                {
                        return std::make_unique<tregression>(parameters);
                }
                virtual rloss_t clone() const override final
                {
                        return std::make_unique<tregression>(*this);
                }

                // compute the error value
                virtual scalar_t error(const vector_t& targets, const vector_t& scores) const override final;

                // compute the loss value & derivatives
                virtual scalar_t value(const vector_t& targets, const vector_t& scores) const override final;
                virtual vector_t vgrad(const vector_t& targets, const vector_t& scores) const override final;

                // predict label indices
                virtual indices_t labels(const vector_t& scores) const override final;
        };

        template <typename tvalue_op, typename tvgrad_op>
        scalar_t regression_loss_t<tvalue_op, tvgrad_op>::error(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());

                return (targets - scores).array().abs().sum();
        }

        template <typename tvalue_op, typename tvgrad_op>
        scalar_t regression_loss_t<tvalue_op, tvgrad_op>::value(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());

                return tvalue_op()(targets, scores);
        }

        template <typename tvalue_op, typename tvgrad_op>
        vector_t regression_loss_t<tvalue_op, tvgrad_op>::vgrad(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());

                return tvgrad_op()(targets, scores);
        }

        template <typename tvalue_op, typename tvgrad_op>
        indices_t regression_loss_t<tvalue_op, tvgrad_op>::labels(const vector_t&) const
        {
                return indices_t();
        }
}

