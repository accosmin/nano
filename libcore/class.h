#pragma once

#include "arch.h"
#include "tensor.h"

namespace ncv
{
        ///
        /// \brief target value of the positive class
        ///
        NANOCV_PUBLIC scalar_t pos_target();

        ///
        /// \brief target value of the negative class
        ///
        NANOCV_PUBLIC scalar_t neg_target();
        
        ///
        /// \brief check if a target value maps to a positive class
        ///
        NANOCV_PUBLIC bool is_pos_target(const scalar_t target);

        ///
        /// \brief target value for multi-class single-label classification problems with [n_labels] classes
        ///
        NANOCV_PUBLIC vector_t class_target(const size_t ilabel, const size_t n_labels);
}
