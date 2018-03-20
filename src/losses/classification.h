#pragma once

#include "loss.h"
#include "cortex.h"
#include <cassert>

namespace nano
{
        ///
        /// \brief multi-class classification loss that predicts the labels with positive outputs.
        ///
        template <typename top>
        class mclassification_t final : public loss_t
        {
        public:
                scalar_t error(const vector_cmap_t& targets, const vector_cmap_t& outputs) const final;
                scalar_t value(const vector_cmap_t& targets, const vector_cmap_t& outputs) const final;
                void vgrad(const vector_cmap_t& targets, const vector_cmap_t& outputs, vector_map_t&&) const final;
        };

        template <typename top>
        scalar_t mclassification_t<top>::error(const vector_cmap_t& targets, const vector_cmap_t& outputs) const
        {
                assert(targets.size() == outputs.size());

                const auto edges = targets.array() * outputs.array();
                const auto epsilon = std::numeric_limits<scalar_t>::epsilon();
                return static_cast<scalar_t>((edges < epsilon).count());
        }

        template <typename top>
        scalar_t mclassification_t<top>::value(const vector_cmap_t& targets, const vector_cmap_t& outputs) const
        {
                assert(targets.size() == outputs.size());

                return top::value(targets, outputs);
        }

        template <typename top>
        void mclassification_t<top>::vgrad(const vector_cmap_t& targets, const vector_cmap_t& outputs, vector_map_t&& ret) const
        {
                assert(targets.size() == outputs.size());

                ret = top::vgrad(targets, outputs);
        }

        ///
        /// \brief single-class classification loss that predicts the label with the highest score.
        ///
        template <typename top>
        class sclassification_t final : public loss_t
        {
        public:
                scalar_t error(const vector_cmap_t& targets, const vector_cmap_t& outputs) const final;
                scalar_t value(const vector_cmap_t& targets, const vector_cmap_t& outputs) const final;
                void vgrad(const vector_cmap_t& targets, const vector_cmap_t& outputs, vector_map_t&&) const final;
        };

        template <typename top>
        scalar_t sclassification_t<top>::error(const vector_cmap_t& targets, const vector_cmap_t& outputs) const
        {
                assert(targets.size() == outputs.size());

                vector_t::Index idx;
                outputs.maxCoeff(&idx);

                return is_pos_target(targets(idx)) ? 0 : 1;
        }

        template <typename top>
        scalar_t sclassification_t<top>::value(const vector_cmap_t& targets, const vector_cmap_t& outputs) const
        {
                assert(targets.size() == outputs.size());

                return top::value(targets, outputs);
        }

        template <typename top>
        void sclassification_t<top>::vgrad(const vector_cmap_t& targets, const vector_cmap_t& outputs, vector_map_t&& ret) const
        {
                assert(targets.size() == outputs.size());

                ret = top::vgrad(targets, outputs);
        }
}
