#include "utest.h"
#include "tensor/index.h"

using namespace nano;

NANO_BEGIN_MODULE(test_tensor_index)

NANO_CASE(index1d)
{
        const auto dims = nano::make_dims(7);

        NANO_CHECK_EQUAL(std::get<0>(dims), 7);
        NANO_CHECK_EQUAL(nano::size(dims), 7);

        NANO_CHECK_EQUAL(nano::index(dims, 0), 0);
        NANO_CHECK_EQUAL(nano::index(dims, 1), 1);
        NANO_CHECK_EQUAL(nano::index(dims, 6), 6);

        NANO_CHECK_EQUAL(nano::index0(dims), nano::index(dims, 0));
        NANO_CHECK_EQUAL(nano::index0(dims, 6), nano::index(dims, 6));

        NANO_CHECK_EQUAL(nano::dims0(dims), nano::make_dims(7));
}

NANO_CASE(index2d)
{
        const auto dims = nano::make_dims(7, 5);

        NANO_CHECK_EQUAL(dims, nano::cat_dims(7, nano::make_dims(5)));

        NANO_CHECK_EQUAL(std::get<0>(dims), 7);
        NANO_CHECK_EQUAL(std::get<1>(dims), 5);
        NANO_CHECK_EQUAL(nano::size(dims), 35);

        NANO_CHECK_EQUAL(nano::index(dims, 0, 1), 1);
        NANO_CHECK_EQUAL(nano::index(dims, 0, 4), 4);
        NANO_CHECK_EQUAL(nano::index(dims, 1, 0), 5);
        NANO_CHECK_EQUAL(nano::index(dims, 3, 2), 17);
        NANO_CHECK_EQUAL(nano::index(dims, 6, 4), 34);

        NANO_CHECK_EQUAL(nano::index0(dims), nano::index(dims, 0, 0));
        NANO_CHECK_EQUAL(nano::index0(dims, 3), nano::index(dims, 3, 0));
        NANO_CHECK_EQUAL(nano::index0(dims, 3, 1), nano::index(dims, 3, 1));

        NANO_CHECK_EQUAL(nano::dims0(dims), nano::make_dims(7, 5));
        NANO_CHECK_EQUAL(nano::dims0(dims, 3), nano::make_dims(5));
}

NANO_CASE(index3d)
{
        const auto dims = nano::make_dims(3, 7, 5);

        NANO_CHECK_EQUAL(dims, nano::cat_dims(3, nano::make_dims(7, 5)));

        NANO_CHECK_EQUAL(std::get<0>(dims), 3);
        NANO_CHECK_EQUAL(std::get<1>(dims), 7);
        NANO_CHECK_EQUAL(std::get<2>(dims), 5);
        NANO_CHECK_EQUAL(nano::size(dims), 105);

        NANO_CHECK_EQUAL(nano::index(dims, 0, 0, 1), 1);
        NANO_CHECK_EQUAL(nano::index(dims, 0, 0, 4), 4);
        NANO_CHECK_EQUAL(nano::index(dims, 0, 1, 0), 5);
        NANO_CHECK_EQUAL(nano::index(dims, 0, 2, 1), 11);
        NANO_CHECK_EQUAL(nano::index(dims, 1, 2, 1), 46);
        NANO_CHECK_EQUAL(nano::index(dims, 1, 0, 3), 38);
        NANO_CHECK_EQUAL(nano::index(dims, 2, 4, 1), 91);
        NANO_CHECK_EQUAL(nano::index(dims, 2, 6, 4), 104);

        NANO_CHECK_EQUAL(nano::index0(dims), nano::index(dims, 0, 0, 0));
        NANO_CHECK_EQUAL(nano::index0(dims, 2), nano::index(dims, 2, 0, 0));
        NANO_CHECK_EQUAL(nano::index0(dims, 2, 4), nano::index(dims, 2, 4, 0));
        NANO_CHECK_EQUAL(nano::index0(dims, 2, 4, 3), nano::index(dims, 2, 4, 3));

        NANO_CHECK_EQUAL(nano::dims0(dims), nano::make_dims(3, 7, 5));
        NANO_CHECK_EQUAL(nano::dims0(dims, 2), nano::make_dims(7, 5));
        NANO_CHECK_EQUAL(nano::dims0(dims, 2, 4), nano::make_dims(5));
}

NANO_END_MODULE()
