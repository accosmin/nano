#pragma once

#include <functional>

namespace nano
{
        ///
        /// \brief combine the current hash with the given value.
        ///
        template <typename tvalue, typename thasher>
        inline void hash_combine(std::size_t& seed, const tvalue& v, thasher& hasher)
        {
                seed ^= hasher(v) + 0x9E3779B97F4A7C15 + (seed << 6) + (seed >> 2);
        }

        template <class tvalue>
        inline void hash_combine(std::size_t& seed, const tvalue& v)
        {
                std::hash<tvalue> hasher;
                hash_combine(seed, v, hasher);
        }

        ///
        /// \brief combine the current hash with the given [begin, end) range.
        ///
        template <class titerator, typename thasher>
        inline void hash_combine_range(std::size_t& seed, titerator begin, const titerator end, thasher& hasher)
        {
                for ( ; begin != end; ++ begin)
                {
                        hash_combine(seed, *begin, hasher);
                }
        }

        template <class titerator>
        inline void hash_combine_range(std::size_t& seed, titerator begin, const titerator end)
        {
                using tnoref = typename std::remove_reference<decltype(*begin)>::type;
                using tvalue = typename std::remove_const<tnoref>::type;

                std::hash<tvalue> hasher;
                hash_combine_range(seed, begin, end, hasher);
        }

        ///
        /// \brief hash the given [begin, end) range.
        ///
        template <class titerator, typename thasher>
        inline std::size_t hash_range(titerator begin, const titerator end, thasher& hasher)
        {
                std::size_t seed = 0;
                hash_combine_range(seed, begin, end, hasher);
                return seed;
        }

        template <class titerator>
        inline std::size_t hash_range(titerator begin, const titerator end)
        {
                using tnoref = typename std::remove_reference<decltype(*begin)>::type;
                using tvalue = typename std::remove_const<tnoref>::type;

                std::hash<tvalue> hasher;
                return hash_range(begin, end, hasher);
        }
}
