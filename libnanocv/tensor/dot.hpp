#pragma once

#include "vector.hpp"

namespace ncv
{
        namespace tensor
        {
                ///
                /// \brief Eigen-based dot product
                ///
                template
                <
                        typename tscalar
                >
                tscalar dot_eig(const tscalar* vec1, const tscalar* vec2, int size)
                {
                        return tensor::make_vector(vec1, size).dot(tensor::make_vector(vec2, size));
                }
        }
}
