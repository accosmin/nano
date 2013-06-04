#ifndef NANOCV_RANDOM_H
#define NANOCV_RANDOM_H

#include <random>
#include <type_traits>

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // uniform random number generator in the [min, max] range.
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        template
        <
                typename trange
        >
        class random_t
        {
        public:
                
                // constructor
                random_t(trange min, trange max)
                        :       m_gen(0),//std::random_device()),
                                m_die(std::min(min, max),
                                      std::max(min, max))
                {
                        std::random_device rd;
                        m_gen = gen_t(rd());
                }
                
                // generate a random value
                trange operator()()
                {
                        return m_die(m_gen);
                }
                
                // fill the [begin, end) range with random values
                template
                <
                        class titerator
                >
                void operator()(titerator begin, titerator end)
                {
                        for (; begin != end; ++ begin)
                        {
                                *begin = operator()();
                        }
                }

                // access functions
                trange min() const { return m_die.min(); }
                trange max() const { return m_die.max(); }
                
        private:

                typedef std::mt19937                                    gen_t;

                typedef typename std::conditional
                <
                        std::is_arithmetic<trange>::value &&
                        std::is_integral<trange>::value,
                        std::uniform_int_distribution<trange>,
                        std::uniform_real_distribution<trange>
                >::type                                                 die_t;

                // attributes
                gen_t           m_gen;
                die_t           m_die;
        };
}

#endif // NANOCV_RANDOM_H

