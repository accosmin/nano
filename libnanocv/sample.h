#pragma once

#include "rect.h"
#include "protocol.h"

namespace ncv
{
        ///
        /// \brief fold: <fold index, protocol: train|test>
        ///
        typedef std::pair<size_t, protocol>     fold_t;
        typedef std::vector<fold_t>             folds_t;

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

        typedef std::vector<sample_t>           samples_t;
        typedef samples_t::iterator             samples_it_t;
        typedef samples_t::const_iterator       samples_const_it_t;

        ///
        /// \brief collect the distinct labels of the given samples
        ///
        strings_t labels(const samples_t& samples);
        strings_t labels(samples_const_it_t begin, samples_const_it_t end);

        ///
        /// \brief compare two samples (to order them for fast caching)
        ///
        inline bool operator<(const sample_t& one, const sample_t& two)
        {
                return one.m_index < two.m_index;
        }
}

