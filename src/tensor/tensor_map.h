#pragma once

#include "storage.h"

namespace nano
{
        ///
        /// \brief 3+D tensor mapping a non-constant array of 2D planes.
        ///
        template <typename tscalar, std::size_t tdimensions, typename tmap = Eigen::Map<tensor_vector_t<tscalar>>>
        struct tensor_map_t : public tensor_storage_t<tmap, tdimensions>
        {
                using tbase = tensor_storage_t<tmap, tdimensions>;

                using tdims = typename tbase::tdims;
                using Index = typename tbase::Index;
                using Scalar = typename tbase::Scalar;

                ///
                /// \brief constructor
                ///
                tensor_map_t() = default;

                ///
                /// \brief constructor
                ///
                tensor_map_t(tscalar* data, const tdims& dims) : tbase(map_vector(data, nano::size(dims)), dims) {}
        };

        ///
        /// \brief 3+D tensor mapping a constant array of 2D planes.
        ///
        template <typename tscalar, std::size_t tdimensions, typename tmap = Eigen::Map<const tensor_vector_t<tscalar>>>
        struct tensor_const_map_t : public tensor_const_storage_t<tmap, tdimensions>
        {
                using tbase = tensor_const_storage_t<tmap, tdimensions>;

                using tdims = typename tbase::tdims;
                using Index = typename tbase::Index;
                using Scalar = typename tbase::Scalar;

                ///
                /// \brief constructor
                ///
                tensor_const_map_t() = default;

                ///
                /// \brief constructor
                ///
                tensor_const_map_t(const tscalar* data, const tdims& dims) : tbase(map_vector(data, nano::size(dims)), dims) {}

                ///
                /// \brief constructor
                ///
                tensor_const_map_t(tscalar* data, const tdims& dims) : tensor_const_map_t(const_cast<const tscalar*>(data), dims) {}
        };
}
