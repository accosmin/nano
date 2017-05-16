#pragma once

#include "tensor.h"

namespace nano
{
        ///
        /// \brief target value of the positive class
        ///
        inline scalar_t pos_target() { return +1; }

        ///
        /// \brief target value of the negative class
        ///
        inline scalar_t neg_target() { return -1; }

        ///
        /// \brief check if a target value maps to a positive class
        ///
        inline bool is_pos_target(const scalar_t target) { return target > 0; }

        ///
        /// \brief target value for multi-class single-label classification problems with [n_labels] classes
        ///
        inline vector_t class_target(const tensor_size_t ilabel, const tensor_size_t n_labels)
        {
                vector_t target = vector_t::Constant(n_labels, neg_target());
                if (ilabel < n_labels)
                {
                        target(ilabel) = pos_target();
                }
                return target;
        }

        ///
        /// \brief target value for multi-class multi-label classification problems based on the sign of the target
        ///
        inline vector_t class_target(const vector_t& scores)
        {
                vector_t target(scores.size());
                for (auto i = 0; i < scores.size(); ++ i)
                {
                        target(i) = is_pos_target(scores(i)) ? pos_target() : neg_target();
                }
                return target;
        }
}
