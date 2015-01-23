#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_dot"

#include <boost/test/unit_test.hpp>
#include "nanocv/types.h"
#include "util/dot.hpp"
#include "util/math.hpp"
#include "tensor/dot.hpp"

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

                const scalar_t dot    = test_dot(ncv::dot<scalar_t>, vec1, vec2);
                const scalar_t dotul2 = test_dot(ncv::dot_unroll<scalar_t, 2>, vec1, vec2);
                const scalar_t dotul3 = test_dot(ncv::dot_unroll<scalar_t, 3>, vec1, vec2);
                const scalar_t dotul4 = test_dot(ncv::dot_unroll<scalar_t, 4>, vec1, vec2);
                const scalar_t dotul5 = test_dot(ncv::dot_unroll<scalar_t, 5>, vec1, vec2);
                const scalar_t dotul6 = test_dot(ncv::dot_unroll<scalar_t, 6>, vec1, vec2);
                const scalar_t dotul7 = test_dot(ncv::dot_unroll<scalar_t, 7>, vec1, vec2);
                const scalar_t dotul8 = test_dot(ncv::dot_unroll<scalar_t, 8>, vec1, vec2);
                const scalar_t doteig = test_dot(ncv::tensor::dot_eig<scalar_t>, vec1, vec2);

                const scalar_t epsilon = 1e-10;

                BOOST_CHECK_LE(math::abs(dot - dot), epsilon);
                BOOST_CHECK_LE(math::abs(dot - dotul2), epsilon);
                BOOST_CHECK_LE(math::abs(dot - dotul3), epsilon);
                BOOST_CHECK_LE(math::abs(dot - dotul4), epsilon);
                BOOST_CHECK_LE(math::abs(dot - dotul5), epsilon);
                BOOST_CHECK_LE(math::abs(dot - dotul6), epsilon);
                BOOST_CHECK_LE(math::abs(dot - dotul7), epsilon);
                BOOST_CHECK_LE(math::abs(dot - dotul8), epsilon);
                BOOST_CHECK_LE(math::abs(dot - doteig), epsilon);
        }
}

BOOST_AUTO_TEST_CASE(test_dot)
{
        using namespace ncv;

        static const size_t min_size = 32 * 1024;
        static const size_t max_size = 4 * 1024 * 1024;

        for (size_t size = min_size; size <= max_size; size *= 2)
        {
                test::test_dot(size);
        }
}

