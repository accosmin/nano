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

NANO_CASE(decode)
{
        const nano::string_t json = R"XXX(
{
        nodes:
        {
                norm0:          {id:norm, type:plane},
                conv0:          {id:conv3d, omaps:128, krows:7, kcols:7, kconn:1, kdrow:2, kdcol:1},
                act0:           {id:act-snorm},
                conv1:          {id:conv3d, omaps:256, krows:5, kcols:5, kconn:2, kdrow:1, kdcol:1},
                act1:           {id:act-pwave},
                conv2:          {id:conv3d, omaps:512, krows:3, kcols:3, kconn:4, kdrow:1, kdcol:1},
                act2:           {id:act-sin},
                affine1:        {id:affine, omaps:1024, orows:1, ocols:1},
                act3:           {id:act-tanh},
                output:         {id:affine, omaps:10, orows:1, ocols:1}
        },
        digraph:
        {
                norm0:          conv0,
                conv0:          act0,
                act0:           conv1,
                conv1:          act1,
                act1:           conv2,
                conv2:          act2,
                act2:           affine1,
                affine1:        act3,
                act3:           output
        }
}
)XXX";
}

NANO_END_MODULE()
