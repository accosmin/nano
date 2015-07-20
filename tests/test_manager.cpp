#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_manager"

#include <boost/test/unit_test.hpp>
#include "nanocv/manager.hpp"

namespace test
{
        using namespace ncv;

        class test_clonable_t : public ncv::clonable_t<test_clonable_t>
        {
        public:

                explicit test_clonable_t(const string_t& configuration = string_t())
                        :       ncv::clonable_t<test_clonable_t>(configuration)
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

BOOST_AUTO_TEST_CASE(test_manager)
{
        using namespace ncv;

        typedef ncv::manager_t<test::test_clonable_t> manager_t;

        manager_t manager;

        // empty manager
        BOOST_CHECK_EQUAL(manager.ids().empty(), true);
        BOOST_CHECK_EQUAL(manager.descriptions().empty(), true);

        BOOST_CHECK_EQUAL(manager.has("ds"), false);
        BOOST_CHECK_EQUAL(manager.has("ds1"), false);
        BOOST_CHECK_EQUAL(manager.has("dd"), false);
        BOOST_CHECK_EQUAL(manager.has(""), false);

        const test::test_obj1_clonable_t obj1;
        const test::test_obj2_clonable_t obj2;
        const test::test_obj3_clonable_t obj3;

        const std::string id1 = "obj1";
        const std::string id2 = "obj2";
        const std::string id3 = "obj3";

        // register objects
        BOOST_CHECK_EQUAL(manager.add(id1, obj1), true);
        BOOST_CHECK_EQUAL(manager.add(id2, obj2), true);
        BOOST_CHECK_EQUAL(manager.add(id3, obj3), true);

        // should not be able to register with the same id anymore
        BOOST_CHECK_EQUAL(manager.add(id1, obj1), false);
        BOOST_CHECK_EQUAL(manager.add(id1, obj2), false);
        BOOST_CHECK_EQUAL(manager.add(id1, obj3), false);

        BOOST_CHECK_EQUAL(manager.add(id2, obj1), false);
        BOOST_CHECK_EQUAL(manager.add(id2, obj2), false);
        BOOST_CHECK_EQUAL(manager.add(id2, obj3), false);

        BOOST_CHECK_EQUAL(manager.add(id3, obj1), false);
        BOOST_CHECK_EQUAL(manager.add(id3, obj2), false);
        BOOST_CHECK_EQUAL(manager.add(id3, obj3), false);

        // check retrieval
        BOOST_CHECK_EQUAL(manager.has(id1), true);
        BOOST_CHECK_EQUAL(manager.has(id2), true);
        BOOST_CHECK_EQUAL(manager.has(id3), true);

        BOOST_CHECK_EQUAL(manager.has(id1 + id2), false);
        BOOST_CHECK_EQUAL(manager.has(id2 + id3), false);
        BOOST_CHECK_EQUAL(manager.has(id3 + id1), false);

        BOOST_CHECK_EQUAL(static_cast<bool>(manager.get(id1)), true);
        BOOST_CHECK_EQUAL(static_cast<bool>(manager.get(id2)), true);
        BOOST_CHECK_EQUAL(static_cast<bool>(manager.get(id3)), true);

        BOOST_CHECK_THROW(manager.get(""), std::runtime_error);
        BOOST_CHECK_THROW(manager.get(id1 + id2 + "ddd"), std::runtime_error);
        BOOST_CHECK_THROW(manager.get("not there"), std::runtime_error);
}
