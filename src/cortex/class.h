#pragma once

#include "arch.h"
#include "tensor.h"

namespace zob
{
        ///
        /// \brief target value of the positive class
        ///
        ZOB_PUBLIC scalar_t pos_target();

        ///
        /// \brief target value of the negative class
        ///
        ZOB_PUBLIC scalar_t neg_target();

        ///
        /// \brief check if a target value maps to a positive class
        ///
        ZOB_PUBLIC bool is_pos_target(const scalar_t target);

        ///
        /// \brief target value for multi-class single-label classification problems with [n_labels] classes
        ///
        ZOB_PUBLIC vector_t class_target(const tensor_index_t ilabel, const tensor_size_t n_labels);

        ///
        /// \brief map the given scores to class labels indexed in the range [0, scores.size())
        ///
        indices_t class_labels(const vector_t& scores);
}

