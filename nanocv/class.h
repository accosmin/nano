#pragma once

#include "nanocv/arch.h"
#include "nanocv/tensor.h"

namespace ncv
{
        ///
        /// \brief target value of the positive class
        ///
        inline scalar_t pos_target() { return +1.0; }

        ///
        /// \brief target value of the negative class
        ///
        inline scalar_t neg_target() { return -1.0; }
        
        ///
        /// \brief check if a target value maps to a positive class
        ///
        inline bool is_pos_target(scalar_t target) { return target > 0.5; }

        ///
        /// \brief target value for multi-class single-label classification problems with [n_labels] classes
        ///
        NANOCV_PUBLIC vector_t class_target(size_t ilabel, size_t n_labels);
}
