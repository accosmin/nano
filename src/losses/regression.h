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
                scalar_t error(const vector_cmap_t& targets, const vector_cmap_t& outputs) const final;
                scalar_t value(const vector_cmap_t& targets, const vector_cmap_t& outputs) const final;
                void vgrad(const vector_cmap_t& targets, const vector_cmap_t& outputs, vector_map_t&&) const final;
        };

        template <typename top>
        scalar_t regression_t<top>::error(const vector_cmap_t& targets, const vector_cmap_t& outputs) const
        {
                assert(targets.size() == outputs.size());

                return (targets - outputs).array().abs().sum();
        }

        template <typename top>
        scalar_t regression_t<top>::value(const vector_cmap_t& targets, const vector_cmap_t& outputs) const
        {
                assert(targets.size() == outputs.size());

                return top::value(targets, outputs);
        }

        template <typename top>
        void regression_t<top>::vgrad(const vector_cmap_t& targets, const vector_cmap_t& outputs, vector_map_t&& ret) const
        {
                assert(targets.size() == outputs.size());

                ret = top::vgrad(targets, outputs);
        }
}
