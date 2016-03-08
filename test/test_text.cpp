#include "unit_test.hpp"
#include "text/align.hpp"
#include "text/algorithm.h"
#include "text/from_params.hpp"
#include "text/concatenate.hpp"
#include <list>
#include <set>

ZOB_BEGIN_MODULE(test_text)

ZOB_CASE(contains)
{
        ZOB_CHECK_EQUAL(text::contains("", 't'), false);
        ZOB_CHECK_EQUAL(text::contains("text", 't'), true);
        ZOB_CHECK_EQUAL(text::contains("naNoCv", 't'), false);
        ZOB_CHECK_EQUAL(text::contains("extension", 't'), true);
}

ZOB_CASE(resize)
{
        ZOB_CHECK_EQUAL(text::align("text", 10, text::alignment::left, '='),   "text======");
        ZOB_CHECK_EQUAL(text::align("text", 10, text::alignment::right, '='),  "======text");
        ZOB_CHECK_EQUAL(text::align("text", 10, text::alignment::center, '='), "===text===");
}

ZOB_CASE(split)
{
        const auto tokens = text::split("= -token1 token2 something ", " =-");

        ZOB_REQUIRE(tokens.size() == 3);
        ZOB_CHECK_EQUAL(tokens[0], "token1");
        ZOB_CHECK_EQUAL(tokens[1], "token2");
        ZOB_CHECK_EQUAL(tokens[2], "something");
}

ZOB_CASE(lower)
{
        ZOB_CHECK_EQUAL(text::lower("Token"), "token");
        ZOB_CHECK_EQUAL(text::lower("ToKEN"), "token");
        ZOB_CHECK_EQUAL(text::lower("token"), "token");
        ZOB_CHECK_EQUAL(text::lower("TOKEN"), "token");
        ZOB_CHECK_EQUAL(text::lower(""), "");
}

ZOB_CASE(upper)
{
        ZOB_CHECK_EQUAL(text::upper("Token"), "TOKEN");
        ZOB_CHECK_EQUAL(text::upper("ToKEN"), "TOKEN");
        ZOB_CHECK_EQUAL(text::upper("token"), "TOKEN");
        ZOB_CHECK_EQUAL(text::upper("TOKEN"), "TOKEN");
        ZOB_CHECK_EQUAL(text::upper(""), "");
}

ZOB_CASE(ends_with)
{
        ZOB_CHECK(text::ends_with("ToKeN", ""));
        ZOB_CHECK(text::ends_with("ToKeN", "N"));
        ZOB_CHECK(text::ends_with("ToKeN", "eN"));
        ZOB_CHECK(text::ends_with("ToKeN", "KeN"));
        ZOB_CHECK(text::ends_with("ToKeN", "oKeN"));
        ZOB_CHECK(text::ends_with("ToKeN", "ToKeN"));

        ZOB_CHECK(!text::ends_with("ToKeN", "n"));
        ZOB_CHECK(!text::ends_with("ToKeN", "en"));
        ZOB_CHECK(!text::ends_with("ToKeN", "ken"));
        ZOB_CHECK(!text::ends_with("ToKeN", "oken"));
        ZOB_CHECK(!text::ends_with("ToKeN", "Token"));
}

ZOB_CASE(iends_with)
{
        ZOB_CHECK(text::iends_with("ToKeN", ""));
        ZOB_CHECK(text::iends_with("ToKeN", "N"));
        ZOB_CHECK(text::iends_with("ToKeN", "eN"));
        ZOB_CHECK(text::iends_with("ToKeN", "KeN"));
        ZOB_CHECK(text::iends_with("ToKeN", "oKeN"));
        ZOB_CHECK(text::iends_with("ToKeN", "ToKeN"));

        ZOB_CHECK(text::iends_with("ToKeN", "n"));
        ZOB_CHECK(text::iends_with("ToKeN", "en"));
        ZOB_CHECK(text::iends_with("ToKeN", "ken"));
        ZOB_CHECK(text::iends_with("ToKeN", "oken"));
        ZOB_CHECK(text::iends_with("ToKeN", "Token"));
}

ZOB_CASE(starts_with)
{
        ZOB_CHECK(text::starts_with("ToKeN", ""));
        ZOB_CHECK(text::starts_with("ToKeN", "T"));
        ZOB_CHECK(text::starts_with("ToKeN", "To"));
        ZOB_CHECK(text::starts_with("ToKeN", "ToK"));
        ZOB_CHECK(text::starts_with("ToKeN", "ToKe"));
        ZOB_CHECK(text::starts_with("ToKeN", "ToKeN"));

        ZOB_CHECK(!text::starts_with("ToKeN", "t"));
        ZOB_CHECK(!text::starts_with("ToKeN", "to"));
        ZOB_CHECK(!text::starts_with("ToKeN", "tok"));
        ZOB_CHECK(!text::starts_with("ToKeN", "toke"));
        ZOB_CHECK(!text::starts_with("ToKeN", "Token"));
}

ZOB_CASE(istarts_with)
{
        ZOB_CHECK(text::istarts_with("ToKeN", ""));
        ZOB_CHECK(text::istarts_with("ToKeN", "t"));
        ZOB_CHECK(text::istarts_with("ToKeN", "to"));
        ZOB_CHECK(text::istarts_with("ToKeN", "Tok"));
        ZOB_CHECK(text::istarts_with("ToKeN", "toKe"));
        ZOB_CHECK(text::istarts_with("ToKeN", "ToKeN"));

        ZOB_CHECK(text::istarts_with("ToKeN", "t"));
        ZOB_CHECK(text::istarts_with("ToKeN", "to"));
        ZOB_CHECK(text::istarts_with("ToKeN", "tok"));
        ZOB_CHECK(text::istarts_with("ToKeN", "toke"));
        ZOB_CHECK(text::istarts_with("ToKeN", "Token"));
}

ZOB_CASE(equals)
{
        ZOB_CHECK(!text::equals("ToKeN", ""));
        ZOB_CHECK(!text::equals("ToKeN", "N"));
        ZOB_CHECK(!text::equals("ToKeN", "eN"));
        ZOB_CHECK(!text::equals("ToKeN", "KeN"));
        ZOB_CHECK(!text::equals("ToKeN", "oKeN"));
        ZOB_CHECK(text::equals("ToKeN", "ToKeN"));

        ZOB_CHECK(!text::equals("ToKeN", "n"));
        ZOB_CHECK(!text::equals("ToKeN", "en"));
        ZOB_CHECK(!text::equals("ToKeN", "ken"));
        ZOB_CHECK(!text::equals("ToKeN", "oken"));
        ZOB_CHECK(!text::equals("ToKeN", "Token"));
}

ZOB_CASE(iequals)
{
        ZOB_CHECK(!text::iequals("ToKeN", ""));
        ZOB_CHECK(!text::iequals("ToKeN", "N"));
        ZOB_CHECK(!text::iequals("ToKeN", "eN"));
        ZOB_CHECK(!text::iequals("ToKeN", "KeN"));
        ZOB_CHECK(!text::iequals("ToKeN", "oKeN"));
        ZOB_CHECK(text::iequals("ToKeN", "ToKeN"));

        ZOB_CHECK(!text::iequals("ToKeN", "n"));
        ZOB_CHECK(!text::iequals("ToKeN", "en"));
        ZOB_CHECK(!text::iequals("ToKeN", "ken"));
        ZOB_CHECK(!text::iequals("ToKeN", "oken"));
        ZOB_CHECK(text::iequals("ToKeN", "Token"));
}

ZOB_CASE(to_string)
{
        ZOB_CHECK_EQUAL(text::to_string(1.7), "1.700000");
        ZOB_CHECK_EQUAL(text::to_string(-4.3f), "-4.300000");
        ZOB_CHECK_EQUAL(text::to_string(1), "1");
        ZOB_CHECK_EQUAL(text::to_string(124545), "124545");
}

ZOB_CASE(from_string)
{
        ZOB_CHECK_EQUAL(text::from_string<double>("1.7"), 1.7);
        ZOB_CHECK_EQUAL(text::from_string<float>("-4.3"), -4.3f);
        ZOB_CHECK_EQUAL(text::from_string<short>("1"), 1);
        ZOB_CHECK_EQUAL(text::from_string<long int>("124545"), 124545);
}

ZOB_CASE(replace)
{
        ZOB_CHECK_EQUAL(text::replace("token-", '-', '_'), "token_");
        ZOB_CHECK_EQUAL(text::replace("t-ken-", '-', '_'), "t_ken_");
        ZOB_CHECK_EQUAL(text::replace("-token", '-', '_'), "_token");
        ZOB_CHECK_EQUAL(text::replace("token_", '-', '_'), "token_");
}

ZOB_CASE(concatenate)
{
        ZOB_CHECK_EQUAL(text::concatenate(std::vector<int>({ 1, 2, 3 }), "-"),        "1-2-3");
        ZOB_CHECK_EQUAL(text::concatenate(std::list<int>({ 1, 2, 3 }), "="),          "1=2=3");
        ZOB_CHECK_EQUAL(text::concatenate(std::set<int>({ 1, 2, 3 }), ","),           "1,2,3");
}

ZOB_CASE(from_params)
{
        const auto config = "param1=1.7,param2=3";

        ZOB_CHECK_EQUAL(text::from_params(config, "param1", 2.0), 1.7);
        ZOB_CHECK_EQUAL(text::from_params(config, "param2", 4343), 3);
        ZOB_CHECK_EQUAL(text::from_params(config, "paramx", 2.0), 2.0);
}

ZOB_END_MODULE()
