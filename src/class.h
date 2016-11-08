#pragma once

#include "arch.h"
#include "tensor.h"

namespace nano
{
        ///
        /// \brief target value of the positive class
        ///
        NANO_PUBLIC scalar_t pos_target();

        ///
        /// \brief target value of the negative class
        ///
        NANO_PUBLIC scalar_t neg_target();

        ///
        /// \brief check if a target value maps to a positive class
        ///
        NANO_PUBLIC bool is_pos_target(const scalar_t target);

        ///
        /// \brief target value for multi-class single-label classification problems with [n_labels] classes
        ///
        NANO_PUBLIC vector_t class_target(const tensor_index_t ilabel, const tensor_size_t n_labels);

        ///
        /// \brief target value for multi-class multi-label classification problems based on the sign of the target
        ///
        NANO_PUBLIC vector_t class_target(const vector_t& scores);
}

