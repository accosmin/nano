#include "unit_test.hpp"
#include "cortex/string.h"
#include "cortex/util/manager.hpp"

namespace test
{
        using namespace cortex;

        class test_clonable_t : public cortex::clonable_t<test_clonable_t>
        {
        public:

                explicit test_clonable_t(const string_t& configuration = string_t())
                        :       cortex::clonable_t<test_clonable_t>(configuration)
                {
                }

                virtual ~test_clonable_t()
                {
                }
        };

        class test_obj1_clonable_t : public test_clonable_t
        {
        public:

                NANOCV_MAKE_CLONABLE(test_obj1_clonable_t, "test obj1")

                explicit test_obj1_clonable_t(const string_t& configuration = string_t())
                        :       test_clonable_t(configuration)
                {
                }
        };

        class test_obj2_clonable_t : public test_clonable_t
        {
        public:

                NANOCV_MAKE_CLONABLE(test_obj2_clonable_t, "test obj2")

                explicit test_obj2_clonable_t(const string_t& configuration = string_t())
                        :       test_clonable_t(configuration)
                {
                }
        };

        class test_obj3_clonable_t : public test_clonable_t
        {
        public:

                NANOCV_MAKE_CLONABLE(test_obj3_clonable_t, "test obj3")

                explicit test_obj3_clonable_t(const string_t& configuration = string_t())
                        :       test_clonable_t(configuration)
                {
                }
        };
}

NANOCV_BEGIN_MODULE(test_manager)

NANOCV_CASE(evaluate)
{
        using namespace cortex;

        typedef cortex::manager_t<test::test_clonable_t> manager_t;

        manager_t manager;

        // empty manager
        NANOCV_CHECK_EQUAL(manager.ids().empty(), true);
        NANOCV_CHECK_EQUAL(manager.descriptions().empty(), true);

        NANOCV_CHECK_EQUAL(manager.has("ds"), false);
        NANOCV_CHECK_EQUAL(manager.has("ds1"), false);
        NANOCV_CHECK_EQUAL(manager.has("dd"), false);
        NANOCV_CHECK_EQUAL(manager.has(""), false);

        const test::test_obj1_clonable_t obj1;
        const test::test_obj2_clonable_t obj2;
        const test::test_obj3_clonable_t obj3;

        const std::string id1 = "obj1";
        const std::string id2 = "obj2";
        const std::string id3 = "obj3";

        // register objects
        NANOCV_CHECK_EQUAL(manager.add(id1, obj1), true);
        NANOCV_CHECK_EQUAL(manager.add(id2, obj2), true);
        NANOCV_CHECK_EQUAL(manager.add(id3, obj3), true);

        // should not be able to register with the same id anymore
        NANOCV_CHECK_EQUAL(manager.add(id1, obj1), false);
        NANOCV_CHECK_EQUAL(manager.add(id1, obj2), false);
        NANOCV_CHECK_EQUAL(manager.add(id1, obj3), false);

        NANOCV_CHECK_EQUAL(manager.add(id2, obj1), false);
        NANOCV_CHECK_EQUAL(manager.add(id2, obj2), false);
        NANOCV_CHECK_EQUAL(manager.add(id2, obj3), false);

        NANOCV_CHECK_EQUAL(manager.add(id3, obj1), false);
        NANOCV_CHECK_EQUAL(manager.add(id3, obj2), false);
        NANOCV_CHECK_EQUAL(manager.add(id3, obj3), false);

        // check retrieval
        NANOCV_REQUIRE(manager.has(id1));
        NANOCV_REQUIRE(manager.has(id2));
        NANOCV_REQUIRE(manager.has(id3));

        NANOCV_CHECK_EQUAL(manager.has(id1 + id2), false);
        NANOCV_CHECK_EQUAL(manager.has(id2 + id3), false);
        NANOCV_CHECK_EQUAL(manager.has(id3 + id1), false);

        NANOCV_CHECK_EQUAL(manager.get(id1)->configuration(), obj1.configuration());
        NANOCV_CHECK_EQUAL(manager.get(id2)->configuration(), obj2.configuration());
        NANOCV_CHECK_EQUAL(manager.get(id3)->configuration(), obj3.configuration());

        NANOCV_CHECK_EQUAL(manager.get(id1)->description(), obj1.description());
        NANOCV_CHECK_EQUAL(manager.get(id2)->description(), obj2.description());
        NANOCV_CHECK_EQUAL(manager.get(id3)->description(), obj3.description());

        NANOCV_CHECK_EQUAL(static_cast<bool>(manager.get(id1)), true);
        NANOCV_CHECK_EQUAL(static_cast<bool>(manager.get(id2)), true);
        NANOCV_CHECK_EQUAL(static_cast<bool>(manager.get(id3)), true);

        NANOCV_CHECK_THROW(manager.get(""), std::runtime_error);
        NANOCV_CHECK_THROW(manager.get(id1 + id2 + "ddd"), std::runtime_error);
        NANOCV_CHECK_THROW(manager.get("not there"), std::runtime_error);
}

NANOCV_END_MODULE()
