#pragma once

#include "loss.h"
#include <cassert>

namespace nano
{
        ///
        /// \brief generic (multivariate) regression loss that upper-bounds
        ///     the L1-distance between target and score/output.
        ///
        template <typename top>
        struct regression_t final : public loss_t
        {
                explicit regression_t(const string_t& params = string_t()) : loss_t(params) {}

                virtual void error(const vector_cmap_t& targets, const vector_cmap_t& scores, scalar_t&) const override;
                virtual void value(const vector_cmap_t& targets, const vector_cmap_t& scores, scalar_t&) const override;
                virtual void vgrad(const vector_cmap_t& targets, const vector_cmap_t& scores, vector_map_t&&) const override;
        };

        template <typename top>
        void regression_t<top>::error(const vector_cmap_t& targets, const vector_cmap_t& scores, scalar_t& ret) const
        {
                assert(targets.size() == scores.size());

                ret = (targets - scores).array().abs().sum();
        }

        template <typename top>
        void regression_t<top>::value(const vector_cmap_t& targets, const vector_cmap_t& scores, scalar_t& ret) const
        {
                assert(targets.size() == scores.size());

                ret = top::value(targets, scores);
        }

        template <typename top>
        void regression_t<top>::vgrad(const vector_cmap_t& targets, const vector_cmap_t& scores, vector_map_t&& ret) const
        {
                assert(targets.size() == scores.size());

                ret = top::vgrad(targets, scores);
        }
}
