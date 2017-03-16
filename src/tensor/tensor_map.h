#pragma once

#include "storage.h"

namespace nano
{
        ///
        /// \brief 3+D tensor mapping an array of 2D planes.
        ///
        template <typename tmap, std::size_t tdimensions>
        struct tensor_map_t : public tensor_storage_t<tmap, tdimensions>
        {
                using tbase = tensor_storage_t<tmap, tdimensions>;
                using tdims = typename tbase::tdims;
                using tscalar = typename tbase::tscalar;

                // Eigen compatible
                using Index = tensor_index_t;
                using Scalar = tscalar;

                ///
                /// \brief constructor
                ///
                template <typename... tsizes>
                tensor_map_t(const tmap& map, const tsizes... dims) :
                        tbase(map, dims...)
                {
                }
        };
}
