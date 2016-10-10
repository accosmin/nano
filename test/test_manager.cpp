#include "utest.hpp"
#include "manager.hpp"

using namespace nano;

class test_clonable_t : public nano::clonable_t<test_clonable_t>
{
public:

        explicit test_clonable_t(const string_t& configuration = string_t()) :
                nano::clonable_t<test_clonable_t>(configuration)
        {
        }

        virtual ~test_clonable_t()
        {
        }
};

class test_obj1_clonable_t : public test_clonable_t
{
public:

        NANO_MAKE_CLONABLE(test_obj1_clonable_t, "p1=def1")

        explicit test_obj1_clonable_t(const string_t& configuration = string_t()) :
                test_clonable_t(configuration)
        {
        }
};

class test_obj2_clonable_t : public test_clonable_t
{
public:

        NANO_MAKE_CLONABLE(test_obj2_clonable_t, "p2=def2")

        explicit test_obj2_clonable_t(const string_t& configuration = string_t()) :
                test_clonable_t(configuration)
        {
        }
};

class test_obj3_clonable_t : public test_clonable_t
{
public:

        NANO_MAKE_CLONABLE(test_obj3_clonable_t, "p3=def3")

        explicit test_obj3_clonable_t(const string_t& configuration = string_t()) :
                test_clonable_t(configuration)
        {
        }
};

NANO_BEGIN_MODULE(test_manager)

NANO_CASE(empty)
{
        nano::manager_t<test_clonable_t> manager;

        NANO_CHECK(manager.ids().empty());
        NANO_CHECK(manager.descriptions().empty());

        NANO_CHECK(!manager.has("ds"));
        NANO_CHECK(!manager.has("ds1"));
        NANO_CHECK(!manager.has("dd"));
        NANO_CHECK(!manager.has(""));
}

NANO_CASE(retrieval)
{
        nano::manager_t<test_clonable_t> manager;

        const test_obj1_clonable_t obj1;
        const test_obj2_clonable_t obj2;
        const test_obj3_clonable_t obj3;

        const string_t id1 = "obj1";
        const string_t id2 = "obj2";
        const string_t id3 = "obj3";

        // register objects
        NANO_CHECK(manager.add(id1, "test obj1", obj1));
        NANO_CHECK(manager.add(id2, "test obj2", obj2));
        NANO_CHECK(manager.add(id3, "test obj3", obj3));

        // should not be able to register with the same id anymore
        NANO_CHECK(!manager.add(id1, "", obj1));
        NANO_CHECK(!manager.add(id1, "", obj2));
        NANO_CHECK(!manager.add(id1, "", obj3));

        NANO_CHECK(!manager.add(id2, "", obj1));
        NANO_CHECK(!manager.add(id2, "", obj2));
        NANO_CHECK(!manager.add(id2, "", obj3));

        NANO_CHECK(!manager.add(id3, "", obj1));
        NANO_CHECK(!manager.add(id3, "", obj2));
        NANO_CHECK(!manager.add(id3, "", obj3));

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

