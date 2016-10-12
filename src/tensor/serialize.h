#pragma once

#include "vector.h"

namespace tensor
{
        ///
        /// \brief serialize a tensor (vector, matrix, 3D tensor) to an 1D array
        ///
        template
        <
                typename tscalar,
                typename ttensor
        >
        tscalar* to_array(tscalar* data, const ttensor& t)
        {
                tensor::map_vector(data, t.size()) = tensor::map_vector(t.data(), t.size());
                return data + t.size();
        }

        template
        <
                typename tscalar,
                typename ttensor,
                typename... tothers
        >
        tscalar* to_array(tscalar* data, const ttensor& t, const tothers&... o)
        {
                return to_array(to_array(data, t), o...);
        }

        ///
        /// \brief deserialize a tensor (vector, matrix, 3D tensor) from an 1D array
        ///
        template
        <
                typename tscalar,
                typename ttensor
        >
        const tscalar* from_array(const tscalar* data, ttensor& t)
        {
                tensor::map_vector(t.data(), t.size()) = tensor::map_vector(data, t.size());
                return data + t.size();
        }

        template
        <
                typename tscalar,
                typename ttensor,
                typename... tothers
        >
        const tscalar* from_array(const tscalar* data, ttensor& t, tothers&... o)
        {
                return from_array(from_array(data, t), o...);
        }
}


