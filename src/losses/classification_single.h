#pragma once

#include "loss.h"
#include "class.h"
#include <cassert>

namespace nano
{
        ///
        /// \brief (multiclass) classification loss that predicts the label with the highest score.
        ///
        template <typename top>
        struct classification_single_t : public loss_t
        {
                explicit classification_single_t(const string_t& parameters = string_t()) : loss_t(parameters) {}

                virtual rloss_t clone(const string_t& parameters) const override;
                virtual rloss_t clone() const override;

                virtual scalar_t error(const vector_t& targets, const vector_t& scores) const override;
                virtual scalar_t value(const vector_t& targets, const vector_t& scores) const override;
                virtual vector_t vgrad(const vector_t& targets, const vector_t& scores) const override;

                virtual indices_t labels(const vector_t& scores) const override;
        };

        template <typename top>
        rloss_t classification_single_t<top>::clone(const string_t& parameters) const
        {
                return std::make_unique<classification_single_t<top>>(parameters);
        }

        template <typename top>
        rloss_t classification_single_t<top>::clone() const
        {
                return std::make_unique<classification_single_t<top>>(*this);
        }

        template <typename top>
        scalar_t classification_single_t<top>::error(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());

                vector_t::Index idx;
                scores.maxCoeff(&idx);

                return is_pos_target(targets(idx)) ? 0 : 1;
        }

        template <typename top>
        scalar_t classification_single_t<top>::value(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());

                return top::value(targets, scores);
        }

        template <typename top>
        vector_t classification_single_t<top>::vgrad(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());

                return top::vgrad(targets, scores);
        }

        template <typename top>
        indices_t classification_single_t<top>::labels(const vector_t& scores) const
        {
                vector_t::Index idx;
                scores.maxCoeff(&idx);

                return { static_cast<size_t>(idx) };
        }
}

