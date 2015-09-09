#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_text"

#include <boost/test/unit_test.hpp>
#include "nanocv/text.hpp"
#include <list>
#include <set>

BOOST_AUTO_TEST_CASE(test_text_resize)
{
        using namespace ncv;

        BOOST_CHECK_EQUAL(text::resize("text", 10, align::left, '='),   "text======");
        BOOST_CHECK_EQUAL(text::resize("text", 10, align::right, '='),  "======text");
        BOOST_CHECK_EQUAL(text::resize("text", 10, align::center, '='), "===text===");
}

BOOST_AUTO_TEST_CASE(test_text_split)
{
        using namespace ncv;

        const auto tokens = text::split("= -token1 token2 something ", " =-");

        BOOST_REQUIRE(tokens.size() == 3);
        BOOST_CHECK_EQUAL(tokens[0], "token1");
        BOOST_CHECK_EQUAL(tokens[1], "token2");
        BOOST_CHECK_EQUAL(tokens[2], "something");
}

BOOST_AUTO_TEST_CASE(test_text_lower)
{
        using namespace ncv;

        BOOST_CHECK_EQUAL(text::lower("Token"), "token");
        BOOST_CHECK_EQUAL(text::lower("ToKEN"), "token");
        BOOST_CHECK_EQUAL(text::lower("token"), "token");
        BOOST_CHECK_EQUAL(text::lower("TOKEN"), "token");
        BOOST_CHECK_EQUAL(text::lower(""), "");
}

BOOST_AUTO_TEST_CASE(test_text_upper)
{
        using namespace ncv;

        BOOST_CHECK_EQUAL(text::upper("Token"), "TOKEN");
        BOOST_CHECK_EQUAL(text::upper("ToKEN"), "TOKEN");
        BOOST_CHECK_EQUAL(text::upper("token"), "TOKEN");
        BOOST_CHECK_EQUAL(text::upper("TOKEN"), "TOKEN");
        BOOST_CHECK_EQUAL(text::upper(""), "");
}

BOOST_AUTO_TEST_CASE(test_text_ends_with)
{
        using namespace ncv;

        BOOST_CHECK(text::ends_with("ToKeN", ""));
        BOOST_CHECK(text::ends_with("ToKeN", "N"));
        BOOST_CHECK(text::ends_with("ToKeN", "eN"));
        BOOST_CHECK(text::ends_with("ToKeN", "KeN"));
        BOOST_CHECK(text::ends_with("ToKeN", "oKeN"));
        BOOST_CHECK(text::ends_with("ToKeN", "ToKeN"));

        BOOST_CHECK(!text::ends_with("ToKeN", "n"));
        BOOST_CHECK(!text::ends_with("ToKeN", "en"));
        BOOST_CHECK(!text::ends_with("ToKeN", "ken"));
        BOOST_CHECK(!text::ends_with("ToKeN", "oken"));
        BOOST_CHECK(!text::ends_with("ToKeN", "Token"));
}

BOOST_AUTO_TEST_CASE(test_text_iends_with)
{
        using namespace ncv;

        BOOST_CHECK(text::iends_with("ToKeN", ""));
        BOOST_CHECK(text::iends_with("ToKeN", "N"));
        BOOST_CHECK(text::iends_with("ToKeN", "eN"));
        BOOST_CHECK(text::iends_with("ToKeN", "KeN"));
        BOOST_CHECK(text::iends_with("ToKeN", "oKeN"));
        BOOST_CHECK(text::iends_with("ToKeN", "ToKeN"));

        BOOST_CHECK(text::iends_with("ToKeN", "n"));
        BOOST_CHECK(text::iends_with("ToKeN", "en"));
        BOOST_CHECK(text::iends_with("ToKeN", "ken"));
        BOOST_CHECK(text::iends_with("ToKeN", "oken"));
        BOOST_CHECK(text::iends_with("ToKeN", "Token"));
}

BOOST_AUTO_TEST_CASE(test_text_equals)
{
        using namespace ncv;

        BOOST_CHECK(!text::equals("ToKeN", ""));
        BOOST_CHECK(!text::equals("ToKeN", "N"));
        BOOST_CHECK(!text::equals("ToKeN", "eN"));
        BOOST_CHECK(!text::equals("ToKeN", "KeN"));
        BOOST_CHECK(!text::equals("ToKeN", "oKeN"));
        BOOST_CHECK(text::equals("ToKeN", "ToKeN"));

        BOOST_CHECK(!text::equals("ToKeN", "n"));
        BOOST_CHECK(!text::equals("ToKeN", "en"));
        BOOST_CHECK(!text::equals("ToKeN", "ken"));
        BOOST_CHECK(!text::equals("ToKeN", "oken"));
        BOOST_CHECK(!text::equals("ToKeN", "Token"));
}

BOOST_AUTO_TEST_CASE(test_text_iequals)
{
        using namespace ncv;

        BOOST_CHECK(!text::iequals("ToKeN", ""));
        BOOST_CHECK(!text::iequals("ToKeN", "N"));
        BOOST_CHECK(!text::iequals("ToKeN", "eN"));
        BOOST_CHECK(!text::iequals("ToKeN", "KeN"));
        BOOST_CHECK(!text::iequals("ToKeN", "oKeN"));
        BOOST_CHECK(text::iequals("ToKeN", "ToKeN"));

        BOOST_CHECK(!text::iequals("ToKeN", "n"));
        BOOST_CHECK(!text::iequals("ToKeN", "en"));
        BOOST_CHECK(!text::iequals("ToKeN", "ken"));
        BOOST_CHECK(!text::iequals("ToKeN", "oken"));
        BOOST_CHECK(text::iequals("ToKeN", "Token"));
}

BOOST_AUTO_TEST_CASE(test_text_to_string)
{
        using namespace ncv;

        BOOST_CHECK_EQUAL(text::to_string(1.7), "1.700000");
        BOOST_CHECK_EQUAL(text::to_string(-4.3f), "-4.300000");
        BOOST_CHECK_EQUAL(text::to_string(1), "1");
        BOOST_CHECK_EQUAL(text::to_string(124545), "124545");
}

BOOST_AUTO_TEST_CASE(test_text_from_string)
{
        using namespace ncv;

        BOOST_CHECK_EQUAL(text::from_string<double>("1.7"), 1.7);
        BOOST_CHECK_EQUAL(text::from_string<float>("-4.3"), -4.3f);
        BOOST_CHECK_EQUAL(text::from_string<short>("1"), 1);
        BOOST_CHECK_EQUAL(text::from_string<long int>("124545"), 124545);
}

BOOST_AUTO_TEST_CASE(test_text_replace)
{
        using namespace ncv;

        BOOST_CHECK_EQUAL(text::replace("token-", '-', '_'), "token_");
        BOOST_CHECK_EQUAL(text::replace("t-ken-", '-', '_'), "t_ken_");
        BOOST_CHECK_EQUAL(text::replace("-token", '-', '_'), "_token");
        BOOST_CHECK_EQUAL(text::replace("token_", '-', '_'), "token_");
}

BOOST_AUTO_TEST_CASE(test_text_concatenate)
{
        using namespace ncv;

        BOOST_CHECK_EQUAL(text::concatenate(std::vector<int>({ 1, 2, 3 }), "-"),        "1-2-3");
        BOOST_CHECK_EQUAL(text::concatenate(std::list<int>({ 1, 2, 3 }), "="),          "1=2=3");
        BOOST_CHECK_EQUAL(text::concatenate(std::set<int>({ 1, 2, 3 }), ","),           "1,2,3");
}

BOOST_AUTO_TEST_CASE(test_from_params)
{
        using namespace ncv;

        const auto config = "param1=1.7,param2=3";

        BOOST_CHECK_EQUAL(text::from_params(config, "param1", 2.0), 1.7);
        BOOST_CHECK_EQUAL(text::from_params(config, "param2", 4343), 3);
        BOOST_CHECK_EQUAL(text::from_params(config, "paramx", 2.0), 2.0);
}
