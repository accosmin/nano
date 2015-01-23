#include "nanocv.h"
#include "util/dot.hpp"
#include "tensor/dot.hpp"

using namespace ncv;

ncv::thread_pool_t pool;

template
<
        typename tvector
>
void rand_vector(size_t size, tvector& vector)
{
        vector.resize(size);
        vector.setRandom();
}

template
<
        typename tvector
>
void zero_vector(tvector& vector)
{
        vector.setZero();
}

template
<
        typename top,
        typename tvector,
        typename tscalar = typename tvector::Scalar
>
tscalar test_dot(
        top op, const char* name, const tvector& vec1, const tvector& vec2)
{
        const ncv::timer_t timer;

        const tscalar ret = op(vec1.data(), vec2.data(), vec1.size());
        
        const size_t milis = static_cast<size_t>(timer.miliseconds());
        std::cout << name << "= " << text::resize(text::to_string(milis), 4, align::right) << "ms  ";
        
        return ret;
}

template
<
        typename tscalar
>
void check(tscalar result, tscalar baseline, const char* name)
{
        const tscalar err = math::abs(result - baseline);
        if (!math::almost_equal(err, tscalar(0)))
        {
                std::cout << name << " FAILED (diff = " << err << ")!" << std::endl;
        }
}

void test_dot(size_t size)
{
        vector_t vec1, vec2;
        rand_vector(size, vec1);
        rand_vector(size, vec2);

        const string_t header = text::to_string(size) + ": ";
        std::cout << text::resize(header, 20);

        const scalar_t dot    = test_dot(ncv::dot<scalar_t>, "dot", vec1, vec2);
        const scalar_t dotul2 = test_dot(ncv::dot_unroll<scalar_t, 2>, "dotul2", vec1, vec2);
        const scalar_t dotul3 = test_dot(ncv::dot_unroll<scalar_t, 3>, "dotul3", vec1, vec2);
        const scalar_t dotul4 = test_dot(ncv::dot_unroll<scalar_t, 4>, "dotul4", vec1, vec2);
        const scalar_t dotul5 = test_dot(ncv::dot_unroll<scalar_t, 5>, "dotul5", vec1, vec2);
        const scalar_t dotul6 = test_dot(ncv::dot_unroll<scalar_t, 6>, "dotul6", vec1, vec2);
        const scalar_t dotul7 = test_dot(ncv::dot_unroll<scalar_t, 7>, "dotul7", vec1, vec2);
        const scalar_t dotul8 = test_dot(ncv::dot_unroll<scalar_t, 8>, "dotul8", vec1, vec2);
        const scalar_t doteig = test_dot(ncv::tensor::dot_eig<scalar_t>, "doteig", vec1, vec2);
        std::cout << std::endl;

        check(dot,      dot, "dot");
        check(dotul2,   dot, "dotul2");
        check(dotul3,   dot, "dotul3");
        check(dotul4,   dot, "dotul4");
        check(dotul5,   dot, "dotul5");
        check(dotul6,   dot, "dotul6");
        check(dotul7,   dot, "dotul7");
        check(dotul8,   dot, "dotul8");
        check(doteig,   dot, "doteig");
}

int main(int argc, char* argv[])
{
        static const size_t min_size = 4 * 1024 * 1024;
        static const size_t max_size = 256 * 1024 * 1024;

        for (size_t size = min_size; size <= max_size; size *= 2)
        {
                test_dot(size);
        }

	return EXIT_SUCCESS;
}

