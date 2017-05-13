#pragma once

#include "loss.h"
#include <cassert>

namespace nano
{
        ///
        /// \brief (multivariate) regression loss that upper-bounds the L1-distance between target and score/output.
        ///
        template <typename top>
        struct regression_t final : public loss_t
        {
                explicit regression_t(const string_t& parameters = string_t()) : loss_t(parameters) {}

                virtual scalar_t error(const vector_t& targets, const vector_t& scores) const override;
                virtual scalar_t value(const vector_t& targets, const vector_t& scores) const override;
                virtual vector_t vgrad(const vector_t& targets, const vector_t& scores) const override;

                virtual indices_t labels(const vector_t& scores) const override;
        };

        template <typename top>
        scalar_t regression_t<top>::error(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());

                return (targets - scores).array().abs().sum();
        }

        template <typename top>
        scalar_t regression_t<top>::value(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());

                return top::value(targets, scores);
        }

        template <typename top>
        vector_t regression_t<top>::vgrad(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());

                return top::vgrad(targets, scores);
        }

        template <typename top>
        indices_t regression_t<top>::labels(const vector_t&) const
        {
                return indices_t();
        }
}

