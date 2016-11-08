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
        struct classification_multi_t final : public loss_t
        {
                explicit classification_multi_t(const string_t& parameters = string_t()) : loss_t(parameters) {}

                virtual scalar_t error(const vector_t& targets, const vector_t& scores) const override;
                virtual scalar_t value(const vector_t& targets, const vector_t& scores) const override;
                virtual vector_t vgrad(const vector_t& targets, const vector_t& scores) const override;

                virtual indices_t labels(const vector_t& scores) const override;
        };

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
                indices_t labels;
                for (auto i = 0; i < scores.size(); ++ i)
                {
                        if (is_pos_target(scores(i)))
                        {
                                labels.push_back(static_cast<size_t>(i));
                        }
                }
                return labels;
        }
}

