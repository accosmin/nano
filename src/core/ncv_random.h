#ifndef NANOCV_RANDOM_H
#define NANOCV_RANDOM_H

#include <random>
#include <type_traits>

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // uniform random number generator in the [min, max] range.
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        // FIXME: Can this be expressed in a simpler way?!
        template
        <
                typename T
        >
        struct uniform_distribution
        {
                // only integral types are allowed!
                typedef typename std::is_integral<T>::type
                        uniform_distribution_should_use_integral_type_t;

                typedef std::uniform_int_distribution<T>        type_t;
        };

        template <>
        struct uniform_distribution<float>
        {
                typedef std::uniform_real_distribution<float>   type_t;
        };

        template <>
        struct uniform_distribution<double>
        {
                typedef std::uniform_real_distribution<double>  type_t;
        };

        template <>
        struct uniform_distribution<long double>
        {
                typedef std::uniform_real_distribution<long double>  type_t;
        };

        template
        <
                typename trange
        >
        class random
        {
        public:
                
                // constructor
                random(trange min, trange max)
                        :       m_gen(0),//std::random_device()),
                                m_die(min, max)
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
                template <class TIterator>
                void operator()(TIterator begin, TIterator end)
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
                typedef typename uniform_distribution<trange>::type_t   die_t;

                // attributes
                gen_t           m_gen;
                die_t           m_die;
        };
}

#endif // NANOCV_RANDOM_H

