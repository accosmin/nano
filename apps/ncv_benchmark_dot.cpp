#include "nanocv.h"
#include "util/dot.hpp"
#include "util/tabulator.h"
#include "tensor/dot.hpp"

using namespace ncv;

template
<
        typename top,
        typename tvector,
        typename tscalar = typename tvector::Scalar
>
tscalar test_dot(
        tabulator_t::row_t& row, top op, const tvector& vec1, const tvector& vec2)
{
        const ncv::timer_t timer;

        const tscalar ret = op(vec1.data(), vec2.data(), vec1.size());
        
        row << timer.microseconds();

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

void test_dot(size_t size, tabulator_t::row_t& row)
{
        vector_t vec1(size), vec2(size);
        vec1.setRandom();
        vec2.setRandom();

        const scalar_t dot    = test_dot(row, ncv::dot<scalar_t>, vec1, vec2);
        const scalar_t dotul2 = test_dot(row, ncv::dot_unroll<scalar_t, 2>, vec1, vec2);
        const scalar_t dotul3 = test_dot(row, ncv::dot_unroll<scalar_t, 3>, vec1, vec2);
        const scalar_t dotul4 = test_dot(row, ncv::dot_unroll<scalar_t, 4>, vec1, vec2);
        const scalar_t dotul5 = test_dot(row, ncv::dot_unroll<scalar_t, 5>, vec1, vec2);
        const scalar_t dotul6 = test_dot(row, ncv::dot_unroll<scalar_t, 6>, vec1, vec2);
        const scalar_t dotul7 = test_dot(row, ncv::dot_unroll<scalar_t, 7>, vec1, vec2);
        const scalar_t dotul8 = test_dot(row, ncv::dot_unroll<scalar_t, 8>, vec1, vec2);
        const scalar_t doteig = test_dot(row, ncv::tensor::dot_eig<scalar_t>, vec1, vec2);

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
        static const size_t min_size = 32 * 1024;
        static const size_t max_size = 4 * 1024 * 1024;

        tabulator_t table("size\\dot");

        table.header() << "dot [us]"
                       << "dotul2 [us]"
                       << "dotul3 [us]"
                       << "dotul4 [us]"
                       << "dotul5 [us]"
                       << "dotul6 [us]"
                       << "dotul7 [us]"
                       << "dotul8 [us]"
                       << "doteig [us]";

        for (size_t size = min_size; size <= max_size; size *= 2)
        {
                tabulator_t::row_t& row = table.append(text::to_string(size / 1024) + "K");

                test_dot(size, row);
        }

        table.print(std::cout);

	return EXIT_SUCCESS;
}

