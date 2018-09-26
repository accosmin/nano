#include "utest.h"
#include "core/random.h"

NANO_BEGIN_MODULE(test_core_random)

NANO_CASE(split2)
{
        const auto count = size_t(120);
        const auto value1 = 7;
        const auto value2 = 5;

        const auto percentage_value1 = size_t(60);
        const auto percentage_value2 = size_t(100) - percentage_value1;

        const auto values = nano::split2(count, value1, percentage_value1, value2);

        NANO_REQUIRE_EQUAL(values.size(), count);
        NANO_CHECK_EQUAL(std::count(values.begin(), values.end(), value1), percentage_value1 * count / 100);
        NANO_CHECK_EQUAL(std::count(values.begin(), values.end(), value2), percentage_value2 * count / 100);
}

NANO_CASE(split3)
{
        const auto count = size_t(420);
        const auto value1 = 7;
        const auto value2 = 5;
        const auto value3 = 3;

        const auto percentage_value1 = size_t(60);
        const auto percentage_value2 = size_t(30);
        const auto percentage_value3 = size_t(100) - percentage_value1 - percentage_value2;

        const auto values = nano::split3(count, value1, percentage_value1, value2, percentage_value2, value3);

        NANO_REQUIRE_EQUAL(values.size(), count);
        NANO_CHECK_EQUAL(std::count(values.begin(), values.end(), value1), percentage_value1 * count / 100);
        NANO_CHECK_EQUAL(std::count(values.begin(), values.end(), value2), percentage_value2 * count / 100);
        NANO_CHECK_EQUAL(std::count(values.begin(), values.end(), value3), percentage_value3 * count / 100);
}

NANO_END_MODULE()
