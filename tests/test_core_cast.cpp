#include "utest.h"
#include "core/cast.h"
#include <list>
#include <set>

enum class enum_type
{
        type1,
        type2,
        type3
};

namespace nano
{
        template <>
        enum_map_t<enum_type> enum_string<enum_type>()
        {
                return
                {
                        { enum_type::type1,     "type1" },
        //                { enum_type::type2,     "type2" },
                        { enum_type::type3,     "type3" }
                };
        }
}

NANO_BEGIN_MODULE(test_core_cast)

NANO_CASE(to_string)
{
        NANO_CHECK_EQUAL(nano::to_string(1), "1");
        NANO_CHECK_EQUAL(nano::to_string(124545), "124545");
}

NANO_CASE(from_string)
{
        NANO_CHECK_EQUAL(nano::from_string<short>("1"), 1);
        NANO_CHECK_EQUAL(nano::from_string<long int>("124545"), 124545);
}

NANO_CASE(enum_string)
{
        NANO_CHECK_EQUAL(nano::to_string(enum_type::type1), "type1");
        NANO_CHECK_THROW(nano::to_string(enum_type::type2), std::invalid_argument);
        NANO_CHECK_EQUAL(nano::to_string(enum_type::type3), "type3");

        NANO_CHECK(nano::from_string<enum_type>("type1") == enum_type::type1);
        NANO_CHECK(nano::from_string<enum_type>("type3") == enum_type::type3);

        NANO_CHECK_THROW(nano::from_string<enum_type>("????"), std::invalid_argument);
        NANO_CHECK_THROW(nano::from_string<enum_type>("type"), std::invalid_argument);
        NANO_CHECK_THROW(nano::from_string<enum_type>("type2"), std::invalid_argument);
}

NANO_CASE(join)
{
        NANO_CHECK_EQUAL(nano::join(std::vector<int>({ 1, 2, 3 }), "-", nullptr, nullptr),      "1-2-3");
        NANO_CHECK_EQUAL(nano::join(std::list<int>({ 1, 2, 3 }), "=", nullptr, nullptr),        "1=2=3");
        NANO_CHECK_EQUAL(nano::join(std::set<int>({ 1, 2, 3 }), ",", nullptr, nullptr),         "1,2,3");

        NANO_CHECK_EQUAL(nano::join(std::vector<int>({ 1, 2, 3 }), "-", "{", "}"),              "{1-2-3}");
        NANO_CHECK_EQUAL(nano::join(std::list<int>({ 1, 2, 3 }), "=", "XXX", "XXX"),            "XXX1=2=3XXX");
        NANO_CHECK_EQUAL(nano::join(std::set<int>({ 1, 2, 3 }), ",", nullptr, ")"),             "1,2,3)");
}

NANO_CASE(strcat)
{
        NANO_CHECK_EQUAL(nano::strcat(nano::string_t("str"), "x", 'a', 42, nano::string_t("end")), "strxa42end");
        NANO_CHECK_EQUAL(nano::strcat("str", nano::string_t("x"), 'a', 42, nano::string_t("end")), "strxa42end");
}

NANO_CASE(make_less)
{
        const auto less = nano::make_less_from_string<int>();

        NANO_CHECK_EQUAL(less("1", "2"), true);
        NANO_CHECK_EQUAL(less("2", "1"), false);
        NANO_CHECK_EQUAL(less("x", "1"), true);
        NANO_CHECK_EQUAL(less("2", "x"), true);
}

NANO_CASE(make_greater)
{
        const auto greater = nano::make_greater_from_string<int>();

        NANO_CHECK_EQUAL(greater("1", "2"), false);
        NANO_CHECK_EQUAL(greater("2", "1"), true);
        NANO_CHECK_EQUAL(greater("x", "1"), true);
        NANO_CHECK_EQUAL(greater("2", "x"), true);
}

NANO_END_MODULE()
