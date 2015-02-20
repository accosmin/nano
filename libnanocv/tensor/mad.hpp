#pragma once

#include "vector.hpp"

namespace ncv
{
        namespace tensor
        {
                ///
                /// \brief Eigen-based mad product
                ///
                template
                <
                        typename tscalar
                >
                void mad(const tscalar* idata, tscalar weight, int size, tscalar* odata)
                {
                        tensor::map_vector(odata, size) += tensor::map_vector(idata, size) * weight;
                }
        }
}
