#include "utest.h"
#include "factory.h"

using namespace nano;

struct object1_configurable_t : public configurable_t
{
        explicit object1_configurable_t(const string_t& config = string_t()) :
                configurable_t(config + ",p1=def1") {}
};

struct object2_configurable_t : public configurable_t
{
        explicit object2_configurable_t(const string_t& config = string_t()) :
                configurable_t(config + ",p2=def2") {}
};

struct object3_configurable_t : public configurable_t
{
        explicit object3_configurable_t(const string_t& config = string_t()) :
                configurable_t(config + ",p3=def3") {}
};

NANO_BEGIN_MODULE(test_manager)

NANO_CASE(empty)
{
        factory_t<configurable_t> manager;

        NANO_CHECK(manager.ids().empty());
        NANO_CHECK(manager.descriptions().empty());

        NANO_CHECK(!manager.has("ds"));
        NANO_CHECK(!manager.has("ds1"));
        NANO_CHECK(!manager.has("dd"));
        NANO_CHECK(!manager.has(""));
}

NANO_CASE(retrieval)
{
        factory_t<configurable_t> manager;

        const object1_configurable_t obj1;
        const object2_configurable_t obj2;
        const object3_configurable_t obj3;

        const string_t id1 = "obj1";
        const string_t id2 = "obj2";
        const string_t id3 = "obj3";

        // register objects
        NANO_CHECK(manager.add<object1_configurable_t>(id1, "test obj1"));
        NANO_CHECK(manager.add<object2_configurable_t>(id2, "test obj2"));
        NANO_CHECK(manager.add<object3_configurable_t>(id3, "test obj3"));

        // should not be able to register with the same id anymore
        NANO_CHECK(!manager.add<object1_configurable_t>(id1, ""));
        NANO_CHECK(!manager.add<object1_configurable_t>(id1, ""));
        NANO_CHECK(!manager.add<object1_configurable_t>(id1, ""));

        NANO_CHECK(!manager.add<object2_configurable_t>(id2, ""));
        NANO_CHECK(!manager.add<object2_configurable_t>(id2, ""));
        NANO_CHECK(!manager.add<object2_configurable_t>(id2, ""));

        NANO_CHECK(!manager.add<object3_configurable_t>(id3, ""));
        NANO_CHECK(!manager.add<object3_configurable_t>(id3, ""));
        NANO_CHECK(!manager.add<object3_configurable_t>(id3, ""));

        // check retrieval
        NANO_REQUIRE(manager.has(id1));
        NANO_REQUIRE(manager.has(id2));
        NANO_REQUIRE(manager.has(id3));

        NANO_CHECK(!manager.has(id1 + id2));
        NANO_CHECK(!manager.has(id2 + id3));
        NANO_CHECK(!manager.has(id3 + id1));

        NANO_CHECK_EQUAL(manager.get(id1)->config(), obj1.config());
        NANO_CHECK_EQUAL(manager.get(id2)->config(), obj2.config());
        NANO_CHECK_EQUAL(manager.get(id3)->config(), obj3.config());

        NANO_CHECK_EQUAL(manager.get(id1, "p1=v1")->config(), "p1=v1,p1=def1");
        NANO_CHECK_EQUAL(manager.get(id2, "p2=v2")->config(), "p2=v2,p2=def2");
        NANO_CHECK_EQUAL(manager.get(id3, "p3=v3")->config(), "p3=v3,p3=def3");

        NANO_CHECK(manager.get(id1));
        NANO_CHECK(manager.get(id2));
        NANO_CHECK(manager.get(id3));

        NANO_CHECK_THROW(manager.get(""), std::runtime_error);
        NANO_CHECK_THROW(manager.get(id1 + id2 + "ddd"), std::runtime_error);
        NANO_CHECK_THROW(manager.get("not there"), std::runtime_error);
}

NANO_END_MODULE()

