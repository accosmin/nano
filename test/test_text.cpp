#include "unit_test.hpp"
#include "text/align.hpp"
#include "text/algorithm.h"
#include "text/from_params.hpp"
#include "text/concatenate.hpp"
#include <list>
#include <set>

NANOCV_BEGIN_MODULE(test_text)

NANOCV_CASE(contains)
{
        NANOCV_CHECK_EQUAL(text::contains("", 't'), false);
        NANOCV_CHECK_EQUAL(text::contains("text", 't'), true);
        NANOCV_CHECK_EQUAL(text::contains("naNoCv", 't'), false);
        NANOCV_CHECK_EQUAL(text::contains("extension", 't'), true);
}

NANOCV_CASE(resize)
{
        NANOCV_CHECK_EQUAL(text::align("text", 10, text::alignment::left, '='),   "text======");
        NANOCV_CHECK_EQUAL(text::align("text", 10, text::alignment::right, '='),  "======text");
        NANOCV_CHECK_EQUAL(text::align("text", 10, text::alignment::center, '='), "===text===");
}

NANOCV_CASE(split)
{
        const auto tokens = text::split("= -token1 token2 something ", " =-");

        NANOCV_REQUIRE(tokens.size() == 3);
        NANOCV_CHECK_EQUAL(tokens[0], "token1");
        NANOCV_CHECK_EQUAL(tokens[1], "token2");
        NANOCV_CHECK_EQUAL(tokens[2], "something");
}

NANOCV_CASE(lower)
{
        NANOCV_CHECK_EQUAL(text::lower("Token"), "token");
        NANOCV_CHECK_EQUAL(text::lower("ToKEN"), "token");
        NANOCV_CHECK_EQUAL(text::lower("token"), "token");
        NANOCV_CHECK_EQUAL(text::lower("TOKEN"), "token");
        NANOCV_CHECK_EQUAL(text::lower(""), "");
}

NANOCV_CASE(upper)
{
        NANOCV_CHECK_EQUAL(text::upper("Token"), "TOKEN");
        NANOCV_CHECK_EQUAL(text::upper("ToKEN"), "TOKEN");
        NANOCV_CHECK_EQUAL(text::upper("token"), "TOKEN");
        NANOCV_CHECK_EQUAL(text::upper("TOKEN"), "TOKEN");
        NANOCV_CHECK_EQUAL(text::upper(""), "");
}

NANOCV_CASE(ends_with)
{
        NANOCV_CHECK(text::ends_with("ToKeN", ""));
        NANOCV_CHECK(text::ends_with("ToKeN", "N"));
        NANOCV_CHECK(text::ends_with("ToKeN", "eN"));
        NANOCV_CHECK(text::ends_with("ToKeN", "KeN"));
        NANOCV_CHECK(text::ends_with("ToKeN", "oKeN"));
        NANOCV_CHECK(text::ends_with("ToKeN", "ToKeN"));

        NANOCV_CHECK(!text::ends_with("ToKeN", "n"));
        NANOCV_CHECK(!text::ends_with("ToKeN", "en"));
        NANOCV_CHECK(!text::ends_with("ToKeN", "ken"));
        NANOCV_CHECK(!text::ends_with("ToKeN", "oken"));
        NANOCV_CHECK(!text::ends_with("ToKeN", "Token"));
}

NANOCV_CASE(iends_with)
{
        NANOCV_CHECK(text::iends_with("ToKeN", ""));
        NANOCV_CHECK(text::iends_with("ToKeN", "N"));
        NANOCV_CHECK(text::iends_with("ToKeN", "eN"));
        NANOCV_CHECK(text::iends_with("ToKeN", "KeN"));
        NANOCV_CHECK(text::iends_with("ToKeN", "oKeN"));
        NANOCV_CHECK(text::iends_with("ToKeN", "ToKeN"));

        NANOCV_CHECK(text::iends_with("ToKeN", "n"));
        NANOCV_CHECK(text::iends_with("ToKeN", "en"));
        NANOCV_CHECK(text::iends_with("ToKeN", "ken"));
        NANOCV_CHECK(text::iends_with("ToKeN", "oken"));
        NANOCV_CHECK(text::iends_with("ToKeN", "Token"));
}

NANOCV_CASE(starts_with)
{
        NANOCV_CHECK(text::starts_with("ToKeN", ""));
        NANOCV_CHECK(text::starts_with("ToKeN", "T"));
        NANOCV_CHECK(text::starts_with("ToKeN", "To"));
        NANOCV_CHECK(text::starts_with("ToKeN", "ToK"));
        NANOCV_CHECK(text::starts_with("ToKeN", "ToKe"));
        NANOCV_CHECK(text::starts_with("ToKeN", "ToKeN"));

        NANOCV_CHECK(!text::starts_with("ToKeN", "t"));
        NANOCV_CHECK(!text::starts_with("ToKeN", "to"));
        NANOCV_CHECK(!text::starts_with("ToKeN", "tok"));
        NANOCV_CHECK(!text::starts_with("ToKeN", "toke"));
        NANOCV_CHECK(!text::starts_with("ToKeN", "Token"));
}

NANOCV_CASE(istarts_with)
{
        NANOCV_CHECK(text::istarts_with("ToKeN", ""));
        NANOCV_CHECK(text::istarts_with("ToKeN", "t"));
        NANOCV_CHECK(text::istarts_with("ToKeN", "to"));
        NANOCV_CHECK(text::istarts_with("ToKeN", "Tok"));
        NANOCV_CHECK(text::istarts_with("ToKeN", "toKe"));
        NANOCV_CHECK(text::istarts_with("ToKeN", "ToKeN"));

        NANOCV_CHECK(text::istarts_with("ToKeN", "t"));
        NANOCV_CHECK(text::istarts_with("ToKeN", "to"));
        NANOCV_CHECK(text::istarts_with("ToKeN", "tok"));
        NANOCV_CHECK(text::istarts_with("ToKeN", "toke"));
        NANOCV_CHECK(text::istarts_with("ToKeN", "Token"));
}

NANOCV_CASE(equals)
{
        NANOCV_CHECK(!text::equals("ToKeN", ""));
        NANOCV_CHECK(!text::equals("ToKeN", "N"));
        NANOCV_CHECK(!text::equals("ToKeN", "eN"));
        NANOCV_CHECK(!text::equals("ToKeN", "KeN"));
        NANOCV_CHECK(!text::equals("ToKeN", "oKeN"));
        NANOCV_CHECK(text::equals("ToKeN", "ToKeN"));

        NANOCV_CHECK(!text::equals("ToKeN", "n"));
        NANOCV_CHECK(!text::equals("ToKeN", "en"));
        NANOCV_CHECK(!text::equals("ToKeN", "ken"));
        NANOCV_CHECK(!text::equals("ToKeN", "oken"));
        NANOCV_CHECK(!text::equals("ToKeN", "Token"));
}

NANOCV_CASE(iequals)
{
        NANOCV_CHECK(!text::iequals("ToKeN", ""));
        NANOCV_CHECK(!text::iequals("ToKeN", "N"));
        NANOCV_CHECK(!text::iequals("ToKeN", "eN"));
        NANOCV_CHECK(!text::iequals("ToKeN", "KeN"));
        NANOCV_CHECK(!text::iequals("ToKeN", "oKeN"));
        NANOCV_CHECK(text::iequals("ToKeN", "ToKeN"));

        NANOCV_CHECK(!text::iequals("ToKeN", "n"));
        NANOCV_CHECK(!text::iequals("ToKeN", "en"));
        NANOCV_CHECK(!text::iequals("ToKeN", "ken"));
        NANOCV_CHECK(!text::iequals("ToKeN", "oken"));
        NANOCV_CHECK(text::iequals("ToKeN", "Token"));
}

NANOCV_CASE(to_string)
{
        NANOCV_CHECK_EQUAL(text::to_string(1.7), "1.700000");
        NANOCV_CHECK_EQUAL(text::to_string(-4.3f), "-4.300000");
        NANOCV_CHECK_EQUAL(text::to_string(1), "1");
        NANOCV_CHECK_EQUAL(text::to_string(124545), "124545");
}

NANOCV_CASE(from_string)
{
        NANOCV_CHECK_EQUAL(text::from_string<double>("1.7"), 1.7);
        NANOCV_CHECK_EQUAL(text::from_string<float>("-4.3"), -4.3f);
        NANOCV_CHECK_EQUAL(text::from_string<short>("1"), 1);
        NANOCV_CHECK_EQUAL(text::from_string<long int>("124545"), 124545);
}

NANOCV_CASE(replace)
{
        NANOCV_CHECK_EQUAL(text::replace("token-", '-', '_'), "token_");
        NANOCV_CHECK_EQUAL(text::replace("t-ken-", '-', '_'), "t_ken_");
        NANOCV_CHECK_EQUAL(text::replace("-token", '-', '_'), "_token");
        NANOCV_CHECK_EQUAL(text::replace("token_", '-', '_'), "token_");
}

NANOCV_CASE(concatenate)
{
        NANOCV_CHECK_EQUAL(text::concatenate(std::vector<int>({ 1, 2, 3 }), "-"),        "1-2-3");
        NANOCV_CHECK_EQUAL(text::concatenate(std::list<int>({ 1, 2, 3 }), "="),          "1=2=3");
        NANOCV_CHECK_EQUAL(text::concatenate(std::set<int>({ 1, 2, 3 }), ","),           "1,2,3");
}

NANOCV_CASE(from_params)
{
        const auto config = "param1=1.7,param2=3";

        NANOCV_CHECK_EQUAL(text::from_params(config, "param1", 2.0), 1.7);
        NANOCV_CHECK_EQUAL(text::from_params(config, "param2", 4343), 3);
        NANOCV_CHECK_EQUAL(text::from_params(config, "paramx", 2.0), 2.0);
}

NANOCV_END_MODULE()
