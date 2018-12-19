#include <utest/utest.h>
#include "core/algorithm.h"

UTEST_BEGIN_MODULE(test_core_algorithm)

UTEST_CASE(contains)
{
        UTEST_CHECK_EQUAL(nano::contains("", 't'), false);
        UTEST_CHECK_EQUAL(nano::contains("text", 't'), true);
        UTEST_CHECK_EQUAL(nano::contains("naNoCv", 't'), false);
        UTEST_CHECK_EQUAL(nano::contains("extension", 't'), true);
}

UTEST_CASE(resize)
{
        UTEST_CHECK_EQUAL(nano::align("text", 10, nano::alignment::left, '='),   "text======");
        UTEST_CHECK_EQUAL(nano::align("text", 10, nano::alignment::right, '='),  "======text");
        UTEST_CHECK_EQUAL(nano::align("text", 10, nano::alignment::left, '='),   "text======");
        UTEST_CHECK_EQUAL(nano::align("text", 10, nano::alignment::center, '='), "===text===");
}

UTEST_CASE(split_str)
{
        const auto tokens = nano::split("= -token1 token2 something ", " =-");

        UTEST_REQUIRE(tokens.size() == 3);
        UTEST_CHECK_EQUAL(tokens[0], "token1");
        UTEST_CHECK_EQUAL(tokens[1], "token2");
        UTEST_CHECK_EQUAL(tokens[2], "something");
}

UTEST_CASE(split_char)
{
        const auto tokens = nano::split("= -token1 token2 something ", '-');

        UTEST_REQUIRE(tokens.size() == 2);
        UTEST_CHECK_EQUAL(tokens[0], "= ");
        UTEST_CHECK_EQUAL(tokens[1], "token1 token2 something ");
}

UTEST_CASE(split_none)
{
        const auto tokens = nano::split("= -token1 token2 something ", "@");

        UTEST_REQUIRE(tokens.size() == 1);
        UTEST_CHECK_EQUAL(tokens[0], "= -token1 token2 something ");
}

UTEST_CASE(lower)
{
        UTEST_CHECK_EQUAL(nano::lower("Token"), "token");
        UTEST_CHECK_EQUAL(nano::lower("ToKEN"), "token");
        UTEST_CHECK_EQUAL(nano::lower("token"), "token");
        UTEST_CHECK_EQUAL(nano::lower("TOKEN"), "token");
        UTEST_CHECK_EQUAL(nano::lower(""), "");
}

UTEST_CASE(upper)
{
        UTEST_CHECK_EQUAL(nano::upper("Token"), "TOKEN");
        UTEST_CHECK_EQUAL(nano::upper("ToKEN"), "TOKEN");
        UTEST_CHECK_EQUAL(nano::upper("token"), "TOKEN");
        UTEST_CHECK_EQUAL(nano::upper("TOKEN"), "TOKEN");
        UTEST_CHECK_EQUAL(nano::upper(""), "");
}

UTEST_CASE(ends_with)
{
        UTEST_CHECK(nano::ends_with("ToKeN", ""));
        UTEST_CHECK(nano::ends_with("ToKeN", "N"));
        UTEST_CHECK(nano::ends_with("ToKeN", "eN"));
        UTEST_CHECK(nano::ends_with("ToKeN", "KeN"));
        UTEST_CHECK(nano::ends_with("ToKeN", "oKeN"));
        UTEST_CHECK(nano::ends_with("ToKeN", "ToKeN"));

        UTEST_CHECK(!nano::ends_with("ToKeN", "n"));
        UTEST_CHECK(!nano::ends_with("ToKeN", "en"));
        UTEST_CHECK(!nano::ends_with("ToKeN", "ken"));
        UTEST_CHECK(!nano::ends_with("ToKeN", "oken"));
        UTEST_CHECK(!nano::ends_with("ToKeN", "Token"));
}

UTEST_CASE(iends_with)
{
        UTEST_CHECK(nano::iends_with("ToKeN", ""));
        UTEST_CHECK(nano::iends_with("ToKeN", "N"));
        UTEST_CHECK(nano::iends_with("ToKeN", "eN"));
        UTEST_CHECK(nano::iends_with("ToKeN", "KeN"));
        UTEST_CHECK(nano::iends_with("ToKeN", "oKeN"));
        UTEST_CHECK(nano::iends_with("ToKeN", "ToKeN"));

        UTEST_CHECK(nano::iends_with("ToKeN", "n"));
        UTEST_CHECK(nano::iends_with("ToKeN", "en"));
        UTEST_CHECK(nano::iends_with("ToKeN", "ken"));
        UTEST_CHECK(nano::iends_with("ToKeN", "oken"));
        UTEST_CHECK(nano::iends_with("ToKeN", "Token"));
}

UTEST_CASE(starts_with)
{
        UTEST_CHECK(nano::starts_with("ToKeN", ""));
        UTEST_CHECK(nano::starts_with("ToKeN", "T"));
        UTEST_CHECK(nano::starts_with("ToKeN", "To"));
        UTEST_CHECK(nano::starts_with("ToKeN", "ToK"));
        UTEST_CHECK(nano::starts_with("ToKeN", "ToKe"));
        UTEST_CHECK(nano::starts_with("ToKeN", "ToKeN"));

        UTEST_CHECK(!nano::starts_with("ToKeN", "t"));
        UTEST_CHECK(!nano::starts_with("ToKeN", "to"));
        UTEST_CHECK(!nano::starts_with("ToKeN", "tok"));
        UTEST_CHECK(!nano::starts_with("ToKeN", "toke"));
        UTEST_CHECK(!nano::starts_with("ToKeN", "Token"));
}

UTEST_CASE(istarts_with)
{
        UTEST_CHECK(nano::istarts_with("ToKeN", ""));
        UTEST_CHECK(nano::istarts_with("ToKeN", "t"));
        UTEST_CHECK(nano::istarts_with("ToKeN", "to"));
        UTEST_CHECK(nano::istarts_with("ToKeN", "Tok"));
        UTEST_CHECK(nano::istarts_with("ToKeN", "toKe"));
        UTEST_CHECK(nano::istarts_with("ToKeN", "ToKeN"));

        UTEST_CHECK(nano::istarts_with("ToKeN", "t"));
        UTEST_CHECK(nano::istarts_with("ToKeN", "to"));
        UTEST_CHECK(nano::istarts_with("ToKeN", "tok"));
        UTEST_CHECK(nano::istarts_with("ToKeN", "toke"));
        UTEST_CHECK(nano::istarts_with("ToKeN", "Token"));
}

UTEST_CASE(equals)
{
        UTEST_CHECK(!nano::equals("ToKeN", ""));
        UTEST_CHECK(!nano::equals("ToKeN", "N"));
        UTEST_CHECK(!nano::equals("ToKeN", "eN"));
        UTEST_CHECK(!nano::equals("ToKeN", "KeN"));
        UTEST_CHECK(!nano::equals("ToKeN", "oKeN"));
        UTEST_CHECK(nano::equals("ToKeN", "ToKeN"));

        UTEST_CHECK(!nano::equals("ToKeN", "n"));
        UTEST_CHECK(!nano::equals("ToKeN", "en"));
        UTEST_CHECK(!nano::equals("ToKeN", "ken"));
        UTEST_CHECK(!nano::equals("ToKeN", "oken"));
        UTEST_CHECK(!nano::equals("ToKeN", "Token"));
}

UTEST_CASE(iequals)
{
        UTEST_CHECK(!nano::iequals("ToKeN", ""));
        UTEST_CHECK(!nano::iequals("ToKeN", "N"));
        UTEST_CHECK(!nano::iequals("ToKeN", "eN"));
        UTEST_CHECK(!nano::iequals("ToKeN", "KeN"));
        UTEST_CHECK(!nano::iequals("ToKeN", "oKeN"));
        UTEST_CHECK(nano::iequals("ToKeN", "ToKeN"));

        UTEST_CHECK(!nano::iequals("ToKeN", "n"));
        UTEST_CHECK(!nano::iequals("ToKeN", "en"));
        UTEST_CHECK(!nano::iequals("ToKeN", "ken"));
        UTEST_CHECK(!nano::iequals("ToKeN", "oken"));
        UTEST_CHECK(nano::iequals("ToKeN", "Token"));
}

UTEST_CASE(replace_str)
{
        UTEST_CHECK_EQUAL(nano::replace("token-", "en-", "_"), "tok_");
        UTEST_CHECK_EQUAL(nano::replace("t-ken-", "ken", "_"), "t-_-");
}

UTEST_CASE(replace_char)
{
        UTEST_CHECK_EQUAL(nano::replace("token-", '-', '_'), "token_");
        UTEST_CHECK_EQUAL(nano::replace("t-ken-", '-', '_'), "t_ken_");
        UTEST_CHECK_EQUAL(nano::replace("-token", '-', '_'), "_token");
        UTEST_CHECK_EQUAL(nano::replace("token_", '-', '_'), "token_");
}

UTEST_END_MODULE()
