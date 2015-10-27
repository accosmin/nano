#pragma once

#include "string.h"
#include "tensor.h"
#include "protocol.h"
#include "vision/rect.h"

namespace cortex
{
        ///
        /// \brief fold: <fold index, protocol: train|test>
        ///
        using fold_t = std::pair<size_t, protocol>;
        using folds_t = std::vector<fold_t>;

        enum class annotation : int
        {
                unlabeled,      ///< un-labeled samples
                annotated       ///< annotated samples
        };

        ///
        /// \brief image-indexed sample
        ///
        struct sample_t
        {
                // constructor
                explicit sample_t(size_t index = 0, coord_t x = 0, coord_t y = 0, coord_t w = 0, coord_t h = 0)
                        :       sample_t(index, rect_t(x, y, w, h))
                {
                }

                explicit sample_t(size_t index, const rect_t& region)
                        :       m_index(index), m_region(region),
                                m_fold({0, protocol::test})
                {
                }

                // check if this sample is annotated
                bool annotated() const { return m_target.size() > 0; }

                // attributes
                size_t          m_index;        ///< image index
                rect_t          m_region;       ///< image coordinates
                string_t        m_label;        ///< label (e.g. classification)
                vector_t        m_target;       ///< target vector to predict
                fold_t          m_fold;
        };

        using samples_t = std::vector<sample_t>;
        using samples_it_t = samples_t::iterator;
        using samples_const_it_t = samples_t::const_iterator;

        ///
        /// \brief collect the distinct labels of the given samples
        ///
        NANOCV_PUBLIC strings_t labels(const samples_t& samples);
        NANOCV_PUBLIC strings_t labels(samples_const_it_t begin, samples_const_it_t end);

        ///
        /// \brief compare two samples (to order them for fast caching)
        ///
        inline bool operator<(const sample_t& one, const sample_t& two)
        {
                return one.m_index < two.m_index;
        }
}

