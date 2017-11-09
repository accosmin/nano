#include "utest.h"
#include "text/json_reader.h"
#include "text/json_writer.h"

NANO_BEGIN_MODULE(test_json)

NANO_CASE(writer_simple)
{
        const size_t tabsize = 0;
        const size_t spacing = 0;
        const bool newline = false;

        auto writer = nano::json_writer_t(tabsize, spacing, newline);
        writer.begin_object();
                writer.pair("param1", "v").next();
                writer.pair("param2", -42).next();
                writer.name("array1").array(1, 2, 3).next();
                writer.pair("param3", +42);
        writer.end_object();

        NANO_CHECK_EQUAL(writer.get(),
        "{\"param1\":\"v\",\"param2\":-42,\"array1\":[1,2,3],\"param3\":42}");
}

NANO_CASE(writer_complex)
{
        const size_t tabsize = 0;
        const size_t spacing = 0;
        const bool newline = false;

        auto writer = nano::json_writer_t(tabsize, spacing, newline);
        writer.begin_object();
                writer.pair("param1", "v").next();
                writer.name("object1").begin_object();
                        writer.name("array1").begin_array();
                                writer.begin_object();
                                        writer.pair("name11", 11).next();
                                        writer.pair("name12", 12);
                                writer.end_object().next();
                                writer.begin_object();
                                        writer.pair("name21", 21).next();
                                        writer.pair("name22", 22);
                                writer.end_object();
                        writer.end_array().next();
                        writer.pair("field1", "v1").next();
                        writer.pair("field2", "v2");
                writer.end_object();
        writer.end_object();

        NANO_CHECK_EQUAL(writer.get(),
        "{\"param1\":\"v\",\"object1\":{\"array1\":[{\"name11\":11,\"name12\":12},{\"name21\":21,\"name22\":22}],"\
        "\"field1\":\"v1\",\"field2\":\"v2\"}}");
}

NANO_CASE(reader_complex)
{
        const nano::string_t json = R"XXX(
{
        "nodes": [
        {
                "name":         "norm0",
                "kind":         "norm",
                "type":         "plane"
        }, {
                "name":         "conv0",
                "kind":         "conv3d",
                "omaps":        128,
                "krows":        7,
                "kcols":        7,
                "kconn":        1,
                "kdrow":        2,
                "kdcol":        1
        }, {
                "name":         "act0",
                "kind":         "act-snorm"
        }, {
                "name":         "conv1",
                "kind":         "conv3d",
                "omaps":        256,
                "krows":        5,
                "kcols":        5,
                "kconn":        2,
                "kdrow":        1,
                "kdcol":        1
        }, {
                "name":         "act1",
                "kind":         "act-pwave"
        }, {
                "name":         "conv2",
                "kind":         "conv3d",
                "omaps":        512,
                "krows":        3,
                "kcols":        3,
                "kconn":        4,
                "kdrow":        1,
                "kdcol":        1
        }, {
                "name":         "act2",
                "kind":         "act-sin"
        }, {
                "name":         "affine1",
                "kind":         "affine",
                "omaps":        1024,
                "orows":        1,
                "ocols":        1
        }, {
                "name":         "act3",
                "kind":         "act-tanh"
        }, {
                "name":         "output",
                "kind":         "affine",
                "omaps":        10,
                "orows":        1,
                "ocols":        1
        }],
        "model": [
                [ "norm0", "conv0", "act0", "conv1", "act1", "conv2", "act2", "affine1", "act3", "output" ]
        ]
}
)XXX";

        using namespace nano;

        const auto callback_begin_object = [] (const char* name, const size_t size)
        {
                std::cout << string_t(name, size) << ": {" << std::endl;
        };
        const auto callback_end_object = [] ()
        {
                std::cout << "}" << std::endl;
        };

        const auto callback_begin_array = [] (const char* name, const size_t size)
        {
                std::cout << string_t(name, size) << ": [" << std::endl;
        };
        const auto callback_end_array = [] ()
        {
                std::cout << "]" << std::endl;
        };

        const auto callback_null = [] (const char* name, const size_t size)
        {
                std::cout << string_t(name, size) << ": null" << std::endl;
        };

        const auto callback_value = [] (const char* name, const size_t size_name, const char* value, const size_t size_value)
        {
                std::cout << string_t(name, size_name) << ": " << string_t(value, size_value) << std::endl;
        };

        json_reader_t reader(json);
        reader.parse(
                callback_begin_object,
                callback_end_object,
                callback_begin_array,
                callback_end_array,
                callback_null,
                callback_value);
}

NANO_END_MODULE()
