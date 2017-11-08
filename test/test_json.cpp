#include "utest.h"
#include "text/json.h"

NANO_BEGIN_MODULE(test_json)

NANO_CASE(encode_object)
{
        nano::string_t json;
        {
                auto encoder = nano::make_json_encoder(json);
                encoder.pair("param1", "v"); encoder.next();
                encoder.pair("param2", -42); encoder.next();
                encoder.pair("param3", +42);
        }

        NANO_CHECK_EQUAL(json, "{param1:v,param2:-42,param3:42}");
}

NANO_CASE(encode_array)
{
        nano::string_t json;
        {
                auto encoder = nano::make_json_encoder(json);
                encoder.pair("param1", "v"); encoder.next();
                {
                        auto array = encoder.array("array");
                        {
                                auto object = array.object("object1");
                                object.pair("name11", 11); object.next();
                                object.pair("name12", 12);
                        } array.next();
                        {
                                auto object = array.object("object2");
                                object.pair("name21", 21); object.next();
                                object.pair("name22", 22);
                        }
                } encoder.next();
                encoder.pair("param3", +42);
        }

        NANO_CHECK_EQUAL(json, "{param1:v,array:[object1:{name11:11,name12:12},object2:{name21:21,name22:22}],param3:42}");
}

NANO_END_MODULE()
