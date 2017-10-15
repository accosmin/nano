#include "utest.h"
#include "text/table.h"
#include "text/config.h"
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
        std::map<enum_type, std::string> enum_string<enum_type>()
        {
                return
                {
                        { enum_type::type1,     "type1" },
        //                { enum_type::type2,     "type2" },
                        { enum_type::type3,     "type3" }
                };
        }
}

template <typename tscalar>
std::ostream& operator<<(std::ostream& os, const std::vector<std::pair<size_t, tscalar>>& values)
{
        os << "{";
        for (const auto& cv : values)
        {
                os << "{" << cv.first << "," << cv.second << "}";
        }
        return os << "}";
}

std::ostream& operator<<(std::ostream& os, const nano::indices_t& indices)
{
        os << "{";
        for (const auto& index : indices)
        {
                os << "{" << index << "}";
        }
        return os << "}";
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
}

NANO_CASE(split)
{
        const auto tokens = nano::split("= -token1 token2 something ", " =-");

        NANO_REQUIRE(tokens.size() == 3);
        NANO_CHECK_EQUAL(tokens[0], "token1");
        NANO_CHECK_EQUAL(tokens[1], "token2");
        NANO_CHECK_EQUAL(tokens[2], "something");
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
        NANO_CHECK_EQUAL(nano::to_string(1.7), "1.700000");
        NANO_CHECK_EQUAL(nano::to_string(-4.3f), "-4.300000");
        NANO_CHECK_EQUAL(nano::to_string(1), "1");
        NANO_CHECK_EQUAL(nano::to_string(124545), "124545");
}

NANO_CASE(from_string)
{
        NANO_CHECK_EQUAL(nano::from_string<double>("1.7"), 1.7);
        NANO_CHECK_EQUAL(nano::from_string<float>("-4.3"), -4.3f);
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

NANO_CASE(replace)
{
        NANO_CHECK_EQUAL(nano::replace("token-", '-', '_'), "token_");
        NANO_CHECK_EQUAL(nano::replace("t-ken-", '-', '_'), "t_ken_");
        NANO_CHECK_EQUAL(nano::replace("-token", '-', '_'), "_token");
        NANO_CHECK_EQUAL(nano::replace("token_", '-', '_'), "token_");
}

NANO_CASE(concatenate)
{
        NANO_CHECK_EQUAL(nano::concatenate(std::vector<int>({ 1, 2, 3 }), "-"),        "1-2-3");
        NANO_CHECK_EQUAL(nano::concatenate(std::list<int>({ 1, 2, 3 }), "="),          "1=2=3");
        NANO_CHECK_EQUAL(nano::concatenate(std::set<int>({ 1, 2, 3 }), ","),           "1,2,3");
}

NANO_CASE(from_params)
{
        const auto config = "param1=1.7,param2=3,param3=-5[-inf,+inf],param4=alpha,param5=beta[description],param6=7";

        NANO_CHECK_EQUAL(nano::from_params(config, "param1", 2.0), 1.7);
        NANO_CHECK_EQUAL(nano::from_params(config, "param2", 42), 3);
        NANO_CHECK_EQUAL(nano::from_params(config, "paramx", 2.4), 2.4);
        NANO_CHECK_EQUAL(nano::from_params(config, "param3", -5), -5);
        NANO_CHECK_EQUAL(nano::from_params(config, "param4", nano::string_t("ccc")), nano::string_t("alpha"));
        NANO_CHECK_EQUAL(nano::from_params(config, "param5", nano::string_t("ddd")), nano::string_t("beta"));
        NANO_CHECK_EQUAL(nano::from_params(config, "param6", 42), 7);
}

NANO_CASE(from_params_no_defaults)
{
        const auto config = "param1=1.7,param2=3,param3=-5[-inf,+inf],param4=alpha,param5=beta[description],param6=7";

        NANO_CHECK_EQUAL(nano::from_params<double>(config, "param1"), 1.7);
        NANO_CHECK_EQUAL(nano::from_params<int>(config, "param2"), 3);
        NANO_CHECK_THROW(nano::from_params<double>(config, "paramx"), std::runtime_error);
        NANO_CHECK_EQUAL(nano::from_params<int>(config, "param3"), -5);
        NANO_CHECK_EQUAL(nano::from_params<nano::string_t>(config, "param4"), nano::string_t("alpha"));
        NANO_CHECK_EQUAL(nano::from_params<nano::string_t>(config, "param5"), nano::string_t("beta"));
        NANO_CHECK_EQUAL(nano::from_params<int>(config, "param6"), 7);
}

NANO_CASE(to_params)
{
        const auto param1 = 7;
        const auto param2 = 42;
        const auto config = nano::to_params("param1", param1, "param2", param2);

        NANO_CHECK_EQUAL(nano::from_params(config, "param1", 34243), param1);
        NANO_CHECK_EQUAL(nano::from_params(config, "param2", 32322), param2);
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

NANO_CASE(table)
{
        nano::table_t t1;
        t1.header() << "head" << "col1" << "col2";
        t1.append() << "row1" << "v11" << "v12";
        t1.append() << "row2" << "v21" << "v22";
        t1.append() << "row3" << "v21" << "v22";

        NANO_CHECK_EQUAL(t1.rows(), 4);
        NANO_CHECK_EQUAL(t1.cols(), 3);

        const auto path = "table.csv";
        const auto delim = ";";

        NANO_CHECK(t1.save(path, delim));

        nano::table_t t2;
        NANO_CHECK(t2.load(path, delim));

        NANO_CHECK_EQUAL(t1, t2);

        std::remove(path);
}

NANO_CASE(table_rows)
{
        using namespace nano;

        nano::table_t table;
        table.header() << "head" << colspan(2) << "colx" << colspan(1) << "col3";
        table.append() << "row1" << "1000" << "9000" << "4000";
        table.append() << "row2" << "3200" << colspan(2) << "2000";
        table.append() << "row3" << colspan(3) << "2500";

        NANO_CHECK_EQUAL(table.rows(), 4);
        NANO_CHECK_EQUAL(table.cols(), 4);

        NANO_CHECK_EQUAL(table.row(0).cols(), 4);
        NANO_CHECK_EQUAL(table.row(1).cols(), 4);
        NANO_CHECK_EQUAL(table.row(2).cols(), 4);
        NANO_CHECK_EQUAL(table.row(3).cols(), 4);

        NANO_CHECK_EQUAL(table.row(0).data(0), "head"); NANO_CHECK_EQUAL(table.row(0).mark(0), "");
        NANO_CHECK_EQUAL(table.row(0).data(1), "colx"); NANO_CHECK_EQUAL(table.row(0).mark(1), "");
        NANO_CHECK_EQUAL(table.row(0).data(2), "colx"); NANO_CHECK_EQUAL(table.row(0).mark(2), "");
        NANO_CHECK_EQUAL(table.row(0).data(3), "col3"); NANO_CHECK_EQUAL(table.row(0).mark(3), "");

        NANO_CHECK_EQUAL(table.row(1).data(0), "row1"); NANO_CHECK_EQUAL(table.row(1).mark(0), "");
        NANO_CHECK_EQUAL(table.row(1).data(1), "1000"); NANO_CHECK_EQUAL(table.row(1).mark(1), "");
        NANO_CHECK_EQUAL(table.row(1).data(2), "9000"); NANO_CHECK_EQUAL(table.row(1).mark(2), "");
        NANO_CHECK_EQUAL(table.row(1).data(3), "4000"); NANO_CHECK_EQUAL(table.row(1).mark(3), "");

        NANO_CHECK_EQUAL(table.row(2).data(0), "row2"); NANO_CHECK_EQUAL(table.row(2).mark(0), "");
        NANO_CHECK_EQUAL(table.row(2).data(1), "3200"); NANO_CHECK_EQUAL(table.row(2).mark(1), "");
        NANO_CHECK_EQUAL(table.row(2).data(2), "2000"); NANO_CHECK_EQUAL(table.row(2).mark(2), "");
        NANO_CHECK_EQUAL(table.row(2).data(3), "2000"); NANO_CHECK_EQUAL(table.row(2).mark(3), "");

        NANO_CHECK_EQUAL(table.row(3).data(0), "row3"); NANO_CHECK_EQUAL(table.row(3).mark(0), "");
        NANO_CHECK_EQUAL(table.row(3).data(1), "2500"); NANO_CHECK_EQUAL(table.row(3).mark(1), "");
        NANO_CHECK_EQUAL(table.row(3).data(2), "2500"); NANO_CHECK_EQUAL(table.row(3).mark(2), "");
        NANO_CHECK_EQUAL(table.row(3).data(3), "2500"); NANO_CHECK_EQUAL(table.row(3).mark(3), "");

        {
                using values_t = std::vector<std::pair<size_t, int>>;
                const auto values0 = table.row(0).collect<int>();
                const auto values1 = table.row(1).collect<int>();
                const auto values2 = table.row(2).collect<int>();
                const auto values3 = table.row(3).collect<int>();
                NANO_CHECK_EQUAL(values0, (values_t{}));
                NANO_CHECK_EQUAL(values1, (values_t{{1, 1000}, {2, 9000}, {3, 4000}}));
                NANO_CHECK_EQUAL(values2, (values_t{{1, 3200}, {2, 2000}, {3, 2000}}));
                NANO_CHECK_EQUAL(values3, (values_t{{1, 2500}, {2, 2500}, {3, 2500}}));
        }
        {
                const auto indices0 = table.row(0).select<int>([] (const auto value) { return value >= 3000; });
                const auto indices1 = table.row(1).select<int>([] (const auto value) { return value >= 3000; });
                const auto indices2 = table.row(2).select<int>([] (const auto value) { return value >= 3000; });
                const auto indices3 = table.row(3).select<int>([] (const auto value) { return value >= 3000; });
                NANO_CHECK_EQUAL(indices0, (indices_t{}));
                NANO_CHECK_EQUAL(indices1, (indices_t{2, 3}));
                NANO_CHECK_EQUAL(indices2, (indices_t{1}));
                NANO_CHECK_EQUAL(indices3, (indices_t{}));
        }
}

NANO_CASE(table_mark)
{
        nano::table_t table;
        table.header() << "name " << "col1" << "col2" << "col3";
        table.append() << "name1" << "1000" << "9000" << "4000";
        table.append() << "name2" << "3200" << "2000" << "5000";
        table.append() << "name3" << "1500" << "7000" << "6000";

        for (size_t r = 0; r < table.rows(); ++ r)
        {
                for (size_t c = 0; c < table.cols(); ++ c)
                {
                        NANO_CHECK_EQUAL(table.row(r).mark(c), "");
                }
        }
        {
                auto tablex = table;
                tablex.mark(nano::make_marker_minimum_col<int>(), "*");

                NANO_CHECK_EQUAL(tablex.row(1).mark(1), "*");
                NANO_CHECK_EQUAL(tablex.row(2).mark(2), "*");
                NANO_CHECK_EQUAL(tablex.row(3).mark(1), "*");
        }
        {
                auto tablex = table;
                tablex.mark(nano::make_marker_maximum_col<int>(), "*");

                NANO_CHECK_EQUAL(tablex.row(1).mark(2), "*");
                NANO_CHECK_EQUAL(tablex.row(2).mark(3), "*");
                NANO_CHECK_EQUAL(tablex.row(3).mark(2), "*");
        }
}

NANO_END_MODULE()
