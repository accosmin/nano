#pragma once

#include "vector.hpp"

namespace tensor
{
        ///
        /// \brief serialize a tensor (vector, matrix, 3D tensor) to an 1D array
        ///
        template
        <
                typename ttensor
        >
        typename ttensor::Scalar* to_array(const ttensor& t, typename ttensor::Scalar* data)
        {
                tensor::map_vector(data, t.size()) = tensor::map_vector(t.data(), t.size());
                return data + t.size();
        }

        ///
        /// \brief deserialize a tensor (vector, matrix, 3D tensor) from an 1D array
        ///
        template
        <
                typename ttensor
        >
        const typename ttensor::Scalar* from_array(ttensor& t, const typename ttensor::Scalar* data)
        {
                tensor::map_vector(t.data(), t.size()) = tensor::map_vector(data, t.size());
                return data + t.size();
        }
}


