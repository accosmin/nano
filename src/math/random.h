#pragma once

#include <random>
#include <limits>
#include <cassert>
#include <type_traits>

namespace nano
{
        ///
        /// \brief uniform random number generator in the [min, max] range.
        ///
        template <typename tscalar, typename = typename std::is_arithmetic<tscalar>::type>
        class random_t
        {
        public:
                using result_type = tscalar;

                ///
                /// \brief constructor
                ///
                explicit random_t(
                        tscalar min = std::numeric_limits<tscalar>::lowest(),
                        tscalar max = std::numeric_limits<tscalar>::max()) :
                        m_gen(std::random_device()()),
                        m_die(std::min(min, max),
                        std::max(min, max))
                {
                }

                ///
                /// \brief generate a random value
                ///
                tscalar operator()()
                {
                        return m_die(m_gen);
                }

                ///
                /// \brief fill the [begin, end) range with random values
                ///
                template <class titerator>
                void operator()(titerator begin, titerator end)
                {
                        for (; begin != end; ++ begin)
                        {
                                *begin = this->operator ()();
                        }
                }

                ///
                /// \brief minimum
                ///
                tscalar min() const { return m_die.min(); }

                ///
                /// \brief maximum
                ///
                tscalar max() const { return m_die.max(); }

        private:

                using gen_t = std::minstd_rand;

                using die_t = typename std::conditional
                <
                        std::is_arithmetic<tscalar>::value &&
                        std::is_integral<tscalar>::value,
                        std::uniform_int_distribution<tscalar>,
                        std::uniform_real_distribution<tscalar>
                >::type;

                // attributes
                gen_t           m_gen;
                die_t           m_die;
        };

        ///
        /// \brief create a random number generator in the given [min, max] range.
        ///
        template <typename tscalar, typename = typename std::is_arithmetic<tscalar>::type>
        random_t<tscalar> make_rng(
                const tscalar min = std::numeric_limits<tscalar>::lowest(),
                const tscalar max = std::numeric_limits<tscalar>::max())
        {
                return random_t<tscalar>(min, max);
        }

        ///
        /// \brief create a random index generator in the given [0, size) range.
        ///
        template <typename tsize, typename = typename std::is_unsigned<tsize>::type>
        random_t<tsize> make_index_rng(const tsize size)
        {
                assert(size > 0);
                return random_t<tsize>(0, size - 1);
        }
}
