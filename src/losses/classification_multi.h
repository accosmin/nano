#pragma once

#include "loss.h"
#include "class.h"
#include <cassert>

namespace nano
{
        ///
        /// \brief (multiclass) classification loss that predicts the labels with positive scores.
        ///
        template <typename top>
        struct classification_multi_t : public loss_t
        {
                explicit classification_multi_t(const string_t& parameters = string_t()) : loss_t(parameters) {}
                virtual ~classification_multi_t() {}

                virtual rloss_t clone(const string_t& parameters) const override final;
                virtual rloss_t clone() const override final;

                virtual scalar_t error(const vector_t& targets, const vector_t& scores) const override final;
                virtual scalar_t value(const vector_t& targets, const vector_t& scores) const override final;
                virtual vector_t vgrad(const vector_t& targets, const vector_t& scores) const override final;

                virtual indices_t labels(const vector_t& scores) const override final;
        };

        template <typename top>
        rloss_t classification_multi_t<top>::clone(const string_t& parameters) const
        {
                return std::make_unique<classification_multi_t<top>>(parameters);
        }

        template <typename top>
        rloss_t classification_multi_t<top>::clone() const
        {
                return std::make_unique<classification_multi_t<top>>(*this);
        }

        template <typename top>
        scalar_t classification_multi_t<top>::error(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());

                const auto edges = targets.array() * scores.array();

                return static_cast<scalar_t>((edges < std::numeric_limits<scalar_t>::epsilon()).count());
        }

        template <typename top>
        scalar_t classification_multi_t<top>::value(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());

                return top::value(targets, scores);
        }

        template <typename top>
        vector_t classification_multi_t<top>::vgrad(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());

                return top::vgrad(targets, scores);
        }

        template <typename top>
        indices_t classification_multi_t<top>::labels(const vector_t& scores) const
        {
                return class_labels(scores);
        }
}

