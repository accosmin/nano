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
                void mad_eig(const tscalar* idata, tscalar weight, int size, tscalar* odata)
                {
                        tensor::make_vector(odata, size) += tensor::make_vector(idata, size) * weight;
                }
        }
}
