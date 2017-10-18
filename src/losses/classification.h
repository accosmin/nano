#pragma once

#include "loss.h"
#include "cortex.h"
#include <cassert>

namespace nano
{
        ///
        /// \brief multi-class classification loss that predicts the labels with positive scores.
        ///
        template <typename top>
        struct mclassification_t final : public loss_t
        {
                explicit mclassification_t(const string_t& params = string_t()) : loss_t(params) {}

                virtual void error(const vector_cmap_t& targets, const vector_cmap_t& scores, scalar_t&) const override;
                virtual void value(const vector_cmap_t& targets, const vector_cmap_t& scores, scalar_t&) const override;
                virtual void vgrad(const vector_cmap_t& targets, const vector_cmap_t& scores, vector_map_t&&) const override;
        };

        template <typename top>
        void mclassification_t<top>::error(const vector_cmap_t& targets, const vector_cmap_t& scores, scalar_t& ret) const
        {
                assert(targets.size() == scores.size());

                const auto edges = targets.array() * scores.array();
                const auto epsilon = std::numeric_limits<scalar_t>::epsilon();
                ret = static_cast<scalar_t>((edges < epsilon).count());
        }

        template <typename top>
        void mclassification_t<top>::value(const vector_cmap_t& targets, const vector_cmap_t& scores, scalar_t& ret) const
        {
                assert(targets.size() == scores.size());

                ret = top::value(targets, scores);
        }

        template <typename top>
        void mclassification_t<top>::vgrad(const vector_cmap_t& targets, const vector_cmap_t& scores, vector_map_t&& ret) const
        {
                assert(targets.size() == scores.size());

                ret = top::vgrad(targets, scores);
        }

        ///
        /// \brief single-class classification loss that predicts the label with the highest score.
        ///
        template <typename top>
        struct sclassification_t final : public loss_t
        {
                explicit sclassification_t(const string_t& params = string_t()) : loss_t(params) {}

                virtual void error(const vector_cmap_t& targets, const vector_cmap_t& scores, scalar_t&) const override;
                virtual void value(const vector_cmap_t& targets, const vector_cmap_t& scores, scalar_t&) const override;
                virtual void vgrad(const vector_cmap_t& targets, const vector_cmap_t& scores, vector_map_t&&) const override;
        };

        template <typename top>
        void sclassification_t<top>::error(const vector_cmap_t& targets, const vector_cmap_t& scores, scalar_t& ret) const
        {
                assert(targets.size() == scores.size());

                vector_t::Index idx;
                scores.maxCoeff(&idx);

                ret = is_pos_target(targets(idx)) ? 0 : 1;
        }

        template <typename top>
        void sclassification_t<top>::value(const vector_cmap_t& targets, const vector_cmap_t& scores, scalar_t& ret) const
        {
                assert(targets.size() == scores.size());

                ret = top::value(targets, scores);
        }

        template <typename top>
        void sclassification_t<top>::vgrad(const vector_cmap_t& targets, const vector_cmap_t& scores, vector_map_t&& ret) const
        {
                assert(targets.size() == scores.size());

                ret = top::vgrad(targets, scores);
        }
}
