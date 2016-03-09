#include "unit_test.hpp"
#include "cortex/util/manager.hpp"

namespace test
{
        class test_clonable_t : public zob::clonable_t<test_clonable_t>
        {
        public:

                explicit test_clonable_t(const std::string& configuration = std::string())
                        :       zob::clonable_t<test_clonable_t>(configuration)
                {
                }

                virtual ~test_clonable_t()
                {
                }
        };

        class test_obj1_clonable_t : public test_clonable_t
        {
        public:

                ZOB_MAKE_CLONABLE(test_obj1_clonable_t, "test obj1")

                explicit test_obj1_clonable_t(const std::string& configuration = std::string())
                        :       test_clonable_t(configuration)
                {
                }
        };

        class test_obj2_clonable_t : public test_clonable_t
        {
        public:

                ZOB_MAKE_CLONABLE(test_obj2_clonable_t, "test obj2")

                explicit test_obj2_clonable_t(const std::string& configuration = std::string())
                        :       test_clonable_t(configuration)
                {
                }
        };

        class test_obj3_clonable_t : public test_clonable_t
        {
        public:

                ZOB_MAKE_CLONABLE(test_obj3_clonable_t, "test obj3")

                explicit test_obj3_clonable_t(const std::string& configuration = std::string())
                        :       test_clonable_t(configuration)
                {
                }
        };
}

ZOB_BEGIN_MODULE(test_manager)

ZOB_CASE(empty)
{
        zob::manager_t<test::test_clonable_t> manager;

        ZOB_CHECK(manager.ids().empty());
        ZOB_CHECK(manager.descriptions().empty());

        ZOB_CHECK(!manager.has("ds"));
        ZOB_CHECK(!manager.has("ds1"));
        ZOB_CHECK(!manager.has("dd"));
        ZOB_CHECK(!manager.has(""));       
}

ZOB_CASE(retrieval)
{
        zob::manager_t<test::test_clonable_t> manager;

        const test::test_obj1_clonable_t obj1;
        const test::test_obj2_clonable_t obj2;
        const test::test_obj3_clonable_t obj3;

        const std::string id1 = "obj1";
        const std::string id2 = "obj2";
        const std::string id3 = "obj3";

        // register objects
        ZOB_CHECK(manager.add(id1, obj1));
        ZOB_CHECK(manager.add(id2, obj2));
        ZOB_CHECK(manager.add(id3, obj3));

        // should not be able to register with the same id anymore
        ZOB_CHECK(!manager.add(id1, obj1));
        ZOB_CHECK(!manager.add(id1, obj2));
        ZOB_CHECK(!manager.add(id1, obj3));

        ZOB_CHECK(!manager.add(id2, obj1));
        ZOB_CHECK(!manager.add(id2, obj2));
        ZOB_CHECK(!manager.add(id2, obj3));

        ZOB_CHECK(!manager.add(id3, obj1));
        ZOB_CHECK(!manager.add(id3, obj2));
        ZOB_CHECK(!manager.add(id3, obj3));

        // check retrieval
        ZOB_REQUIRE(manager.has(id1));
        ZOB_REQUIRE(manager.has(id2));
        ZOB_REQUIRE(manager.has(id3));

        ZOB_CHECK(!manager.has(id1 + id2));
        ZOB_CHECK(!manager.has(id2 + id3));
        ZOB_CHECK(!manager.has(id3 + id1));

        ZOB_CHECK_EQUAL(manager.get(id1)->configuration(), obj1.configuration());
        ZOB_CHECK_EQUAL(manager.get(id2)->configuration(), obj2.configuration());
        ZOB_CHECK_EQUAL(manager.get(id3)->configuration(), obj3.configuration());

        ZOB_CHECK_EQUAL(manager.get(id1)->description(), obj1.description());
        ZOB_CHECK_EQUAL(manager.get(id2)->description(), obj2.description());
        ZOB_CHECK_EQUAL(manager.get(id3)->description(), obj3.description());

        ZOB_CHECK(manager.get(id1));
        ZOB_CHECK(manager.get(id2));
        ZOB_CHECK(manager.get(id3));

        ZOB_CHECK_THROW(manager.get(""), std::runtime_error);
        ZOB_CHECK_THROW(manager.get(id1 + id2 + "ddd"), std::runtime_error);
        ZOB_CHECK_THROW(manager.get("not there"), std::runtime_error);
}

ZOB_END_MODULE()

