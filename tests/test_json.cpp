#include "utest.h"
#include "text/json_reader.h"
#include "text/json_writer.h"

using namespace nano;

std::ostream& operator<<(std::ostream& os, const json_tag tag)
{
        return os << to_string(tag);
}

NANO_BEGIN_MODULE(test_json)

NANO_CASE(writer_simple)
{
        json_writer_t writer;
        writer.new_object();
                writer.pair("param1", "v").next();
                writer.pair("param2", -42).next();
                writer.name("array1").array(1, 2, 3).next();
                writer.pair("param3", +42).next();
                writer.name("null").null();
        writer.end_object();

        NANO_CHECK_EQUAL(writer.str(),
        "{\"param1\":\"v\",\"param2\":-42,\"array1\":[1,2,3],\"param3\":42,\"null\":null}");
}

NANO_CASE(writer_complex)
{
        json_writer_t writer;
        writer.new_object();
                writer.pair("param1", "v").next();
                writer.name("object1").new_object();
                        writer.name("array1").new_array();
                                writer.new_object();
                                        writer.pair("name11", 11).next();
                                        writer.pair("name12", 12);
                                writer.end_object().next();
                                writer.new_object();
                                        writer.pair("name21", 21).next();
                                        writer.pair("name22", 22);
                                writer.end_object();
                        writer.end_array().next();
                        writer.pair("field1", "v1").next();
                        writer.pair("field2", "v2");
                writer.end_object();
        writer.end_object();

        NANO_CHECK_EQUAL(writer.str(),
        "{\"param1\":\"v\",\"object1\":{\"array1\":[{\"name11\":11,\"name12\":12},{\"name21\":21,\"name22\":22}],"\
        "\"field1\":\"v1\",\"field2\":\"v2\"}}");
}

NANO_CASE(writer_complex_with_pairs)
{
        json_writer_t writer;
        writer.new_object();
                writer.pair("param1", "v").next();
                writer.name("object1").new_object();
                        writer.name("array1").new_array();
                                writer.new_object();
                                        writer.pairs("name11", 11, "name12", 12);
                                writer.end_object().next();
                                writer.new_object();
                                        writer.pairs("name21", 21, "name22", 22);
                                writer.end_object();
                        writer.end_array().next();
                        writer.pairs("field1", "v1", "field2", "v2");
                writer.end_object();
        writer.end_object();

        NANO_CHECK_EQUAL(writer.str(),
        "{\"param1\":\"v\",\"object1\":{\"array1\":[{\"name11\":11,\"name12\":12},{\"name21\":21,\"name22\":22}],"\
        "\"field1\":\"v1\",\"field2\":\"v2\"}}");
}

NANO_CASE(reader_object)
{
        const string_t json = R"XXX(
{
        "string":       "str",
        "integer":      42,
        "tag1":         "new_object",
        "tag2":         "value",
        "oups":         "[this,that]"
}
)XXX";

        auto object_str = string_t{};
        auto object_integer = 0;
        auto object_tag1 = json_tag::null;
        auto object_tag2 = json_tag::null;
        auto object_oups = string_t{};
        auto unknown = 1;

        json_reader_t reader(json);
        reader.object(
                "string", object_str, "integer", object_integer, "tag1", object_tag1, "tag2", object_tag2,
                "oups", object_oups, "unknown", unknown);

        NANO_CHECK_EQUAL(object_str, "str");
        NANO_CHECK_EQUAL(object_integer, 42);
        NANO_CHECK_EQUAL(object_tag1, json_tag::new_object);
        NANO_CHECK_EQUAL(object_tag2, json_tag::value);
        NANO_CHECK_EQUAL(object_oups, "[this,that]");
        NANO_CHECK_EQUAL(unknown, 1);

        // all tags should be accounted for
        NANO_CHECK_EQUAL(reader.str(), json);
        NANO_CHECK_EQUAL(reader.pos(), json.size());
        NANO_CHECK_EQUAL(reader.tag(), json_tag::none);
        NANO_CHECK(reader == reader.end());
}

NANO_CASE(reader_simple)
{
        const string_t json = R"XXX(
{
        "string":       "str",
        "integer":      42,
        "tag1":         "new_object",
        "tag2":         "value"
}
)XXX";

        const std::vector<std::pair<string_t, json_tag>> calls = {
                { "", json_tag::new_object },
                { "string", json_tag::name },
                { "str", json_tag::value },
                { "integer", json_tag::name },
                { "42", json_tag::value },
                { "tag1", json_tag::name },
                { "new_object", json_tag::value },
                { "tag2", json_tag::name },
                { "value", json_tag::value },
                { "", json_tag::end_object }
        };

        // iterator version
        {
                json_reader_t reader(json);

                size_t index = 0;
                for (auto itend = reader.end(); reader != itend; ++ reader)
                {
                        const auto token = *reader;
                        const auto name = std::get<0>(token);
                        const auto size = std::get<1>(token);
                        const auto jtag = std::get<2>(token);

                        NANO_REQUIRE_LESS(index, calls.size());
                        NANO_CHECK_EQUAL(calls[index].first, string_t(name, size));
                        NANO_CHECK_EQUAL(calls[index].second, jtag);
                        ++ index;
                }

                // all tags should be accounted for
                NANO_CHECK_EQUAL(index, calls.size());
                NANO_CHECK_EQUAL(reader.str(), json);
                NANO_CHECK_EQUAL(reader.pos(), json.size());
                NANO_CHECK_EQUAL(reader.tag(), json_tag::none);
                NANO_CHECK(reader == reader.end());
        }

        // range-based loop version
        {
                size_t index = 0;
                for (const auto& token : json_reader_t(json))
                {
                        const auto name = std::get<0>(token);
                        const auto size = std::get<1>(token);
                        const auto jtag = std::get<2>(token);

                        NANO_REQUIRE_LESS(index, calls.size());
                        NANO_CHECK_EQUAL(calls[index].first, string_t(name, size));
                        NANO_CHECK_EQUAL(calls[index].second, jtag);
                        ++ index;
                }

                // all tags should be accounted for
                NANO_CHECK_EQUAL(index, calls.size());
        }
}

NANO_CASE(reader_complex)
{

        const string_t json = R"XXX(
{
        "string":       "value1",
        "integer":      42,
        "array":        [1, 2, 3],
        "null":         null,
        "object_array": [
        {
                "name": "name1",
                "int":  1
        },
        {
                "name": "name2",
                "int":  2
        }]
}
)XXX";

        const std::vector<std::pair<string_t, json_tag>> calls = {
                { "", json_tag::new_object },
                { "string", json_tag::name },
                { "value1", json_tag::value },
                { "integer", json_tag::name },
                { "42", json_tag::value },
                { "array", json_tag::name },
                { "", json_tag::new_array },
                        { "1", json_tag::value },
                        { "2", json_tag::value },
                        { "3", json_tag::value },
                { "", json_tag::end_array },
                { "null", json_tag::name },
                { "null", json_tag::null },
                { "object_array", json_tag::name },
                { "", json_tag::new_array },
                { "", json_tag::new_object },
                        { "name", json_tag::name },
                        { "name1", json_tag::value },
                        { "int", json_tag::name },
                        { "1", json_tag::value },
                { "", json_tag::end_object },
                { "", json_tag::new_object },
                        { "name", json_tag::name },
                        { "name2", json_tag::value },
                        { "int", json_tag::name },
                        { "2", json_tag::value },
                { "", json_tag::end_object },
                { "", json_tag::end_array },
                { "", json_tag::end_object }
        };

        // iterator version
        {
                json_reader_t reader(json);

                size_t index = 0;
                for (auto itend = reader.end(); reader != itend; ++ reader)
                {
                        const auto token = *reader;
                        const auto name = std::get<0>(token);
                        const auto size = std::get<1>(token);
                        const auto jtag = std::get<2>(token);

                        NANO_REQUIRE_LESS(index, calls.size());
                        NANO_CHECK_EQUAL(calls[index].first, string_t(name, size));
                        NANO_CHECK_EQUAL(calls[index].second, jtag);
                        ++ index;
                }

                // all tags should be accounted for
                NANO_CHECK_EQUAL(index, calls.size());
                NANO_CHECK_EQUAL(reader.str(), json);
                NANO_CHECK_EQUAL(reader.pos(), json.size());
                NANO_CHECK_EQUAL(reader.tag(), json_tag::none);
                NANO_CHECK(reader == reader.end());
        }

        // range-based loop version
        {
                size_t index = 0;
                for (const auto& token : json_reader_t(json))
                {
                        const auto name = std::get<0>(token);
                        const auto size = std::get<1>(token);
                        const auto jtag = std::get<2>(token);

                        NANO_REQUIRE_LESS(index, calls.size());
                        NANO_CHECK_EQUAL(calls[index].first, string_t(name, size));
                        NANO_CHECK_EQUAL(calls[index].second, jtag);
                        ++ index;
                }

                // all tags should be accounted for
                NANO_CHECK_EQUAL(index, calls.size());
        }
}

NANO_END_MODULE()
