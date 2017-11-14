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
        class regression_t final : public loss_t
        {
        public:
                scalar_t error(const vector_cmap_t& targets, const vector_cmap_t& scores) const override;
                scalar_t value(const vector_cmap_t& targets, const vector_cmap_t& scores) const override;
                void vgrad(const vector_cmap_t& targets, const vector_cmap_t& scores, vector_map_t&&) const override;
        };

        template <typename top>
        scalar_t regression_t<top>::error(const vector_cmap_t& targets, const vector_cmap_t& scores) const
        {
                assert(targets.size() == scores.size());

                return (targets - scores).array().abs().sum();
        }

        template <typename top>
        scalar_t regression_t<top>::value(const vector_cmap_t& targets, const vector_cmap_t& scores) const
        {
                assert(targets.size() == scores.size());

                return top::value(targets, scores);
        }

        template <typename top>
        void regression_t<top>::vgrad(const vector_cmap_t& targets, const vector_cmap_t& scores, vector_map_t&& ret) const
        {
                assert(targets.size() == scores.size());

                ret = top::vgrad(targets, scores);
        }
}
