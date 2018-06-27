#include "utest.h"
#include "text/cast.h"
#include "text/algorithm.h"
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

NANO_BEGIN_MODULE(test_text)

NANO_CASE(contains)
{
        NANO_CHECK_EQUAL(nano::contains("", 't'), false);
        NANO_CHECK_EQUAL(nano::contains("text", 't'), true);
        NANO_CHECK_EQUAL(nano::contains("naNoCv", 't'), false);
        NANO_CHECK_EQUAL(nano::contains("extension", 't'), true);
}

NANO_CASE(resize)
{
        NANO_CHECK_EQUAL(nano::align("text", 10, nano::alignment::left, '='),   "text======");
        NANO_CHECK_EQUAL(nano::align("text", 10, nano::alignment::right, '='),  "======text");
        NANO_CHECK_EQUAL(nano::align("text", 10, nano::alignment::left, '='),   "text======");
        NANO_CHECK_EQUAL(nano::align("text", 10, nano::alignment::center, '='), "===text===");
}

NANO_CASE(split_str)
{
        const auto tokens = nano::split("= -token1 token2 something ", " =-");

        NANO_REQUIRE(tokens.size() == 3);
        NANO_CHECK_EQUAL(tokens[0], "token1");
        NANO_CHECK_EQUAL(tokens[1], "token2");
        NANO_CHECK_EQUAL(tokens[2], "something");
}

NANO_CASE(split_char)
{
        const auto tokens = nano::split("= -token1 token2 something ", '-');

        NANO_REQUIRE(tokens.size() == 2);
        NANO_CHECK_EQUAL(tokens[0], "= ");
        NANO_CHECK_EQUAL(tokens[1], "token1 token2 something ");
}

NANO_CASE(split_none)
{
        const auto tokens = nano::split("= -token1 token2 something ", "@");

        NANO_REQUIRE(tokens.size() == 1);
        NANO_CHECK_EQUAL(tokens[0], "= -token1 token2 something ");
}

NANO_CASE(lower)
{
        NANO_CHECK_EQUAL(nano::lower("Token"), "token");
        NANO_CHECK_EQUAL(nano::lower("ToKEN"), "token");
        NANO_CHECK_EQUAL(nano::lower("token"), "token");
        NANO_CHECK_EQUAL(nano::lower("TOKEN"), "token");
        NANO_CHECK_EQUAL(nano::lower(""), "");
}

NANO_CASE(upper)
{
        NANO_CHECK_EQUAL(nano::upper("Token"), "TOKEN");
        NANO_CHECK_EQUAL(nano::upper("ToKEN"), "TOKEN");
        NANO_CHECK_EQUAL(nano::upper("token"), "TOKEN");
        NANO_CHECK_EQUAL(nano::upper("TOKEN"), "TOKEN");
        NANO_CHECK_EQUAL(nano::upper(""), "");
}

NANO_CASE(ends_with)
{
        NANO_CHECK(nano::ends_with("ToKeN", ""));
        NANO_CHECK(nano::ends_with("ToKeN", "N"));
        NANO_CHECK(nano::ends_with("ToKeN", "eN"));
        NANO_CHECK(nano::ends_with("ToKeN", "KeN"));
        NANO_CHECK(nano::ends_with("ToKeN", "oKeN"));
        NANO_CHECK(nano::ends_with("ToKeN", "ToKeN"));

        NANO_CHECK(!nano::ends_with("ToKeN", "n"));
        NANO_CHECK(!nano::ends_with("ToKeN", "en"));
        NANO_CHECK(!nano::ends_with("ToKeN", "ken"));
        NANO_CHECK(!nano::ends_with("ToKeN", "oken"));
        NANO_CHECK(!nano::ends_with("ToKeN", "Token"));
}

NANO_CASE(iends_with)
{
        NANO_CHECK(nano::iends_with("ToKeN", ""));
        NANO_CHECK(nano::iends_with("ToKeN", "N"));
        NANO_CHECK(nano::iends_with("ToKeN", "eN"));
        NANO_CHECK(nano::iends_with("ToKeN", "KeN"));
        NANO_CHECK(nano::iends_with("ToKeN", "oKeN"));
        NANO_CHECK(nano::iends_with("ToKeN", "ToKeN"));

        NANO_CHECK(nano::iends_with("ToKeN", "n"));
        NANO_CHECK(nano::iends_with("ToKeN", "en"));
        NANO_CHECK(nano::iends_with("ToKeN", "ken"));
        NANO_CHECK(nano::iends_with("ToKeN", "oken"));
        NANO_CHECK(nano::iends_with("ToKeN", "Token"));
}

NANO_CASE(starts_with)
{
        NANO_CHECK(nano::starts_with("ToKeN", ""));
        NANO_CHECK(nano::starts_with("ToKeN", "T"));
        NANO_CHECK(nano::starts_with("ToKeN", "To"));
        NANO_CHECK(nano::starts_with("ToKeN", "ToK"));
        NANO_CHECK(nano::starts_with("ToKeN", "ToKe"));
        NANO_CHECK(nano::starts_with("ToKeN", "ToKeN"));

        NANO_CHECK(!nano::starts_with("ToKeN", "t"));
        NANO_CHECK(!nano::starts_with("ToKeN", "to"));
        NANO_CHECK(!nano::starts_with("ToKeN", "tok"));
        NANO_CHECK(!nano::starts_with("ToKeN", "toke"));
        NANO_CHECK(!nano::starts_with("ToKeN", "Token"));
}

NANO_CASE(istarts_with)
{
        NANO_CHECK(nano::istarts_with("ToKeN", ""));
        NANO_CHECK(nano::istarts_with("ToKeN", "t"));
        NANO_CHECK(nano::istarts_with("ToKeN", "to"));
        NANO_CHECK(nano::istarts_with("ToKeN", "Tok"));
        NANO_CHECK(nano::istarts_with("ToKeN", "toKe"));
        NANO_CHECK(nano::istarts_with("ToKeN", "ToKeN"));

        NANO_CHECK(nano::istarts_with("ToKeN", "t"));
        NANO_CHECK(nano::istarts_with("ToKeN", "to"));
        NANO_CHECK(nano::istarts_with("ToKeN", "tok"));
        NANO_CHECK(nano::istarts_with("ToKeN", "toke"));
        NANO_CHECK(nano::istarts_with("ToKeN", "Token"));
}

NANO_CASE(equals)
{
        NANO_CHECK(!nano::equals("ToKeN", ""));
        NANO_CHECK(!nano::equals("ToKeN", "N"));
        NANO_CHECK(!nano::equals("ToKeN", "eN"));
        NANO_CHECK(!nano::equals("ToKeN", "KeN"));
        NANO_CHECK(!nano::equals("ToKeN", "oKeN"));
        NANO_CHECK(nano::equals("ToKeN", "ToKeN"));

        NANO_CHECK(!nano::equals("ToKeN", "n"));
        NANO_CHECK(!nano::equals("ToKeN", "en"));
        NANO_CHECK(!nano::equals("ToKeN", "ken"));
        NANO_CHECK(!nano::equals("ToKeN", "oken"));
        NANO_CHECK(!nano::equals("ToKeN", "Token"));
}

NANO_CASE(iequals)
{
        NANO_CHECK(!nano::iequals("ToKeN", ""));
        NANO_CHECK(!nano::iequals("ToKeN", "N"));
        NANO_CHECK(!nano::iequals("ToKeN", "eN"));
        NANO_CHECK(!nano::iequals("ToKeN", "KeN"));
        NANO_CHECK(!nano::iequals("ToKeN", "oKeN"));
        NANO_CHECK(nano::iequals("ToKeN", "ToKeN"));

        NANO_CHECK(!nano::iequals("ToKeN", "n"));
        NANO_CHECK(!nano::iequals("ToKeN", "en"));
        NANO_CHECK(!nano::iequals("ToKeN", "ken"));
        NANO_CHECK(!nano::iequals("ToKeN", "oken"));
        NANO_CHECK(nano::iequals("ToKeN", "Token"));
}

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

NANO_CASE(replace_str)
{
        NANO_CHECK_EQUAL(nano::replace("token-", "en-", "_"), "tok_");
        NANO_CHECK_EQUAL(nano::replace("t-ken-", "ken", "_"), "t-_-");
}

NANO_CASE(replace_char)
{
        NANO_CHECK_EQUAL(nano::replace("token-", '-', '_'), "token_");
        NANO_CHECK_EQUAL(nano::replace("t-ken-", '-', '_'), "t_ken_");
        NANO_CHECK_EQUAL(nano::replace("-token", '-', '_'), "_token");
        NANO_CHECK_EQUAL(nano::replace("token_", '-', '_'), "token_");
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

NANO_CASE(filename)
{
        NANO_CHECK_EQUAL(nano::filename("source"), "source");
        NANO_CHECK_EQUAL(nano::filename("source.out"), "source.out");
        NANO_CHECK_EQUAL(nano::filename("a.out.ext"), "a.out.ext");
        NANO_CHECK_EQUAL(nano::filename("/usr/include/awesome"), "awesome");
        NANO_CHECK_EQUAL(nano::filename("/usr/include/awesome.txt"), "awesome.txt");
}

NANO_CASE(extension)
{
        NANO_CHECK_EQUAL(nano::extension("source"), "");
        NANO_CHECK_EQUAL(nano::extension("source.out"), "out");
        NANO_CHECK_EQUAL(nano::extension("a.out.ext"), "ext");
        NANO_CHECK_EQUAL(nano::extension("/usr/include/awesome"), "");
        NANO_CHECK_EQUAL(nano::extension("/usr/include/awesome.txt"), "txt");
}

NANO_CASE(stem)
{
        NANO_CHECK_EQUAL(nano::stem("source"), "source");
        NANO_CHECK_EQUAL(nano::stem("source.out"), "source");
        NANO_CHECK_EQUAL(nano::stem("a.out.ext"), "a.out");
        NANO_CHECK_EQUAL(nano::stem("/usr/include/awesome"), "awesome");
        NANO_CHECK_EQUAL(nano::stem("/usr/include/awesome.txt"), "awesome");
}

NANO_CASE(dirname)
{
        NANO_CHECK_EQUAL(nano::dirname("source"), "./");
        NANO_CHECK_EQUAL(nano::dirname("source.out"), "./");
        NANO_CHECK_EQUAL(nano::dirname("a.out.ext"), "./");
        NANO_CHECK_EQUAL(nano::dirname("/usr/include/awesome"), "/usr/include/");
        NANO_CHECK_EQUAL(nano::dirname("/usr/include/awesome.txt"), "/usr/include/");
}

NANO_END_MODULE()
