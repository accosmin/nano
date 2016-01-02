#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_text"

#include <boost/test/unit_test.hpp>
#include "text/filesystem.h"

BOOST_AUTO_TEST_CASE(test_filename)
{
        BOOST_CHECK_EQUAL(text::filename("source"), "source");
        BOOST_CHECK_EQUAL(text::filename("source.out"), "source.out");
        BOOST_CHECK_EQUAL(text::filename("a.out.ext"), "a.out.ext");
        BOOST_CHECK_EQUAL(text::filename("/usr/include/awesome"), "awesome");
        BOOST_CHECK_EQUAL(text::filename("/usr/include/awesome.txt"), "awesome.txt");
}

BOOST_AUTO_TEST_CASE(test_extension)
{
        BOOST_CHECK_EQUAL(text::extension("source"), "");
        BOOST_CHECK_EQUAL(text::extension("source.out"), "out");
        BOOST_CHECK_EQUAL(text::extension("a.out.ext"), "ext");
        BOOST_CHECK_EQUAL(text::extension("/usr/include/awesome"), "");
        BOOST_CHECK_EQUAL(text::extension("/usr/include/awesome.txt"), "txt");
}

BOOST_AUTO_TEST_CASE(test_stem)
{
        BOOST_CHECK_EQUAL(text::stem("source"), "source");
        BOOST_CHECK_EQUAL(text::stem("source.out"), "source");
        BOOST_CHECK_EQUAL(text::stem("a.out.ext"), "a.out");
        BOOST_CHECK_EQUAL(text::stem("/usr/include/awesome"), "awesome");
        BOOST_CHECK_EQUAL(text::stem("/usr/include/awesome.txt"), "awesome");
}

BOOST_AUTO_TEST_CASE(test_dirname)
{
        BOOST_CHECK_EQUAL(text::dirname("source"), "./");
        BOOST_CHECK_EQUAL(text::dirname("source.out"), "./");
        BOOST_CHECK_EQUAL(text::dirname("a.out.ext"), "./");
        BOOST_CHECK_EQUAL(text::dirname("/usr/include/awesome"), "/usr/include/");
        BOOST_CHECK_EQUAL(text::dirname("/usr/include/awesome.txt"), "/usr/include/");
}
