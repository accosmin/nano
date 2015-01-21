#include "nanocv.h"
#include "util/dot.hpp"
#include "util/mad.hpp"

using namespace ncv;

ncv::thread_pool_t pool;

typedef ncv::scalar_t test_scalar_t;
typedef ncv::tensor::vector_types_t<test_scalar_t>::tvector     test_vector_t;
typedef ncv::tensor::vector_types_t<test_scalar_t>::tvectors    test_vectors_t;

namespace ncv
{
        template
        <
                typename tscalar
        >
        tscalar dot_eig(const tscalar* vec1, const tscalar* vec2, int size)
        {
                return tensor::make_vector(vec1, size).dot(tensor::make_vector(vec2, size));
        }

        template
        <
                typename tscalar
        >
        void mad_eig(const tscalar* idata, tscalar weight, int size, tscalar* odata)
        {
                tensor::make_vector(odata, size) += tensor::make_vector(idata, size) * weight;
        }
}

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
        typename top,
        typename tvector,
        typename tscalar = typename tvector::Scalar
>
tscalar test_mad(
        top op, const char* name, const tvector& vec1, const tvector& vec2, tscalar wei)
{
        vector_t cvec1 = vec1;
        vector_t cvec2 = vec2;

        const ncv::timer_t timer;

        op(cvec1.data(), wei, cvec1.size(), cvec2.data());

        const size_t milis = static_cast<size_t>(timer.miliseconds());
        std::cout << name << "= " << text::resize(text::to_string(milis), 4, align::right) << "ms  ";

        return cvec2.sum();
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
        test_vector_t vec1, vec2;
        rand_vector(size, vec1);
        rand_vector(size, vec2);

        const string_t header = text::to_string(size) + ": ";
        std::cout << text::resize(header, 20);

        const test_scalar_t dot    = test_dot(ncv::dot<test_scalar_t>, "dot", vec1, vec2);
        const test_scalar_t dotul2 = test_dot(ncv::dot_unroll<test_scalar_t, 2>, "dotul2", vec1, vec2);
        const test_scalar_t dotul3 = test_dot(ncv::dot_unroll<test_scalar_t, 3>, "dotul3", vec1, vec2);
        const test_scalar_t dotul4 = test_dot(ncv::dot_unroll<test_scalar_t, 4>, "dotul4", vec1, vec2);
        const test_scalar_t dotul5 = test_dot(ncv::dot_unroll<test_scalar_t, 5>, "dotul5", vec1, vec2);
        const test_scalar_t dotul6 = test_dot(ncv::dot_unroll<test_scalar_t, 6>, "dotul6", vec1, vec2);
        const test_scalar_t dotul7 = test_dot(ncv::dot_unroll<test_scalar_t, 7>, "dotul7", vec1, vec2);
        const test_scalar_t dotul8 = test_dot(ncv::dot_unroll<test_scalar_t, 8>, "dotul8", vec1, vec2);
        const test_scalar_t doteig = test_dot(ncv::dot_eig<test_scalar_t>, "doteig", vec1, vec2);
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

void test_mad(size_t size)
{
        test_vector_t vec1, vec2;
        rand_vector(size, vec1);
        rand_vector(size, vec2);

        test_scalar_t wei;

        wei = vec1(0) + vec2(3);

        const string_t header = text::to_string(size) + ": ";
        std::cout << text::resize(header, 20);

        const test_scalar_t mad    = test_mad(ncv::mad<test_scalar_t>, "mad", vec1, vec2, wei);
        const test_scalar_t madul2 = test_mad(ncv::mad_unroll<test_scalar_t, 2>, "madul2", vec1, vec2, wei);
        const test_scalar_t madul3 = test_mad(ncv::mad_unroll<test_scalar_t, 3>, "madul3", vec1, vec2, wei);
        const test_scalar_t madul4 = test_mad(ncv::mad_unroll<test_scalar_t, 4>, "madul4", vec1, vec2, wei);
        const test_scalar_t madul5 = test_mad(ncv::mad_unroll<test_scalar_t, 5>, "madul5", vec1, vec2, wei);
        const test_scalar_t madul6 = test_mad(ncv::mad_unroll<test_scalar_t, 6>, "madul6", vec1, vec2, wei);
        const test_scalar_t madul7 = test_mad(ncv::mad_unroll<test_scalar_t, 7>, "madul7", vec1, vec2, wei);
        const test_scalar_t madul8 = test_mad(ncv::mad_unroll<test_scalar_t, 8>, "madul8", vec1, vec2, wei);
        const test_scalar_t madeig = test_mad(ncv::mad_eig<test_scalar_t>, "madeig", vec1, vec2, wei);
        std::cout << std::endl;

        check(mad,      mad, "mad");
        check(madul2,   mad, "madul2");
        check(madul3,   mad, "madul3");
        check(madul4,   mad, "madul4");
        check(madul5,   mad, "madul5");
        check(madul6,   mad, "madul6");
        check(madul7,   mad, "madul7");
        check(madul8,   mad, "madul8");
        check(madeig,   mad, "madeig");
}

int main(int argc, char* argv[])
{
        static const size_t min_size = 4 * 1024 * 1024;
        static const size_t max_size = 256 * 1024 * 1024;

        for (size_t size = min_size; size <= max_size; size *= 2)
        {
                test_dot(size);
        }
        std::cout << std::endl;

        for (size_t size = min_size; size <= max_size; size *= 2)
        {
                test_mad(size);
        }

	return EXIT_SUCCESS;
}

