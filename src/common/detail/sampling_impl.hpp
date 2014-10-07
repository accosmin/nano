#pragma once

#include <vector>
#include <algorithm>
#include "common/random.hpp"

namespace ncv
{
        namespace detail
        {
                ///
                /// \brief create a random subset of indices
                ///
                inline std::vector<std::size_t> uniform_indices(std::size_t capacity, std::size_t size)
                {
                        random_t<std::size_t> rng(1, capacity);

                        std::vector<std::size_t> indices(size);
                        for (std::size_t i = 0; i < size; i ++)
                        {
                                indices[i] = rng() - 1;
                        }

                        std::sort(indices.begin(), indices.end());

                        return indices;
                }
        }
}
