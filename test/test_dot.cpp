#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_dot"

#include <boost/test/unit_test.hpp>
#include "nanocv/dot.hpp"
#include "nanocv/tensor.h"
#include "nanocv/math/abs.hpp"
#include "nanocv/math/epsilon.hpp"
#include "nanocv/tensor/dot.hpp"

namespace test
{
        using namespace ncv;

        template
        <
                typename top,
                typename tvector,
                typename tscalar = typename tvector::Scalar
        >
        tscalar test_dot(top op, const tvector& vec1, const tvector& vec2)
        {
                return op(vec1.data(), vec2.data(), vec1.size());
        }

        void test_dot(size_t size)
        {
                vector_t vec1(size), vec2(size);
                vec1.setRandom();
                vec2.setRandom();

                vec1.array() *= scalar_t(1) / std::cbrt(scalar_t(size));
                vec2.array() *= scalar_t(1) / std::cbrt(scalar_t(size));

                const scalar_t dot    = test_dot(ncv::dot<scalar_t>, vec1, vec2);
                const scalar_t dotul2 = test_dot(ncv::dot_unroll<scalar_t, 2>, vec1, vec2);
                const scalar_t dotul3 = test_dot(ncv::dot_unroll<scalar_t, 3>, vec1, vec2);
                const scalar_t dotul4 = test_dot(ncv::dot_unroll<scalar_t, 4>, vec1, vec2);
                const scalar_t dotul5 = test_dot(ncv::dot_unroll<scalar_t, 5>, vec1, vec2);
                const scalar_t dotul6 = test_dot(ncv::dot_unroll<scalar_t, 6>, vec1, vec2);
                const scalar_t dotul7 = test_dot(ncv::dot_unroll<scalar_t, 7>, vec1, vec2);
                const scalar_t dotul8 = test_dot(ncv::dot_unroll<scalar_t, 8>, vec1, vec2);
                const scalar_t doteig = test_dot(ncv::tensor::dot<scalar_t>, vec1, vec2);

                BOOST_CHECK_LE(math::abs(dot - dot), math::epsilon1<scalar_t>());
                BOOST_CHECK_LE(math::abs(dot - dotul2), math::epsilon1<scalar_t>());
                BOOST_CHECK_LE(math::abs(dot - dotul3), math::epsilon1<scalar_t>());
                BOOST_CHECK_LE(math::abs(dot - dotul4), math::epsilon1<scalar_t>());
                BOOST_CHECK_LE(math::abs(dot - dotul5), math::epsilon1<scalar_t>());
                BOOST_CHECK_LE(math::abs(dot - dotul6), math::epsilon1<scalar_t>());
                BOOST_CHECK_LE(math::abs(dot - dotul7), math::epsilon1<scalar_t>());
                BOOST_CHECK_LE(math::abs(dot - dotul8), math::epsilon1<scalar_t>());
                BOOST_CHECK_LE(math::abs(dot - doteig), math::epsilon1<scalar_t>());
        }
}

BOOST_AUTO_TEST_CASE(test_dot)
{
        using namespace ncv;

        static const size_t min_size = 1024;
        static const size_t max_size = 1024 * 1024;

        for (size_t size = min_size; size <= max_size; size *= 2)
        {
                test::test_dot(size);
        }
}

