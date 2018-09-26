#include "utest.h"
#include "core/algorithm.h"

NANO_BEGIN_MODULE(test_core_algorithm)

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

NANO_END_MODULE()
