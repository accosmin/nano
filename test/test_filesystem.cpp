#include "utest.hpp"
#include "text/filesystem.h"

NANO_BEGIN_MODULE(test_filesystem)

NANO_CASE(filename)
{
        NANO_CHECK_EQUAL(nano::filename("source"), "source");
        NANO_CHECK_EQUAL(nano::filename("source.out"), "source.out");
        NANO_CHECK_EQUAL(nano::filename("a.out.ext"), "a.out.ext");
        NANO_CHECK_EQUAL(nano::filename("/usr/include/awesome"), "awesome");
        NANO_CHECK_EQUAL(nano::filename("/usr/include/awesome.txt"), "awesome.txt");
}

NANO_CASE(extension)
{
        NANO_CHECK_EQUAL(nano::extension("source"), "");
        NANO_CHECK_EQUAL(nano::extension("source.out"), "out");
        NANO_CHECK_EQUAL(nano::extension("a.out.ext"), "ext");
        NANO_CHECK_EQUAL(nano::extension("/usr/include/awesome"), "");
        NANO_CHECK_EQUAL(nano::extension("/usr/include/awesome.txt"), "txt");
}

NANO_CASE(stem)
{
        NANO_CHECK_EQUAL(nano::stem("source"), "source");
        NANO_CHECK_EQUAL(nano::stem("source.out"), "source");
        NANO_CHECK_EQUAL(nano::stem("a.out.ext"), "a.out");
        NANO_CHECK_EQUAL(nano::stem("/usr/include/awesome"), "awesome");
        NANO_CHECK_EQUAL(nano::stem("/usr/include/awesome.txt"), "awesome");
}

NANO_CASE(dirname)
{
        NANO_CHECK_EQUAL(nano::dirname("source"), "./");
        NANO_CHECK_EQUAL(nano::dirname("source.out"), "./");
        NANO_CHECK_EQUAL(nano::dirname("a.out.ext"), "./");
        NANO_CHECK_EQUAL(nano::dirname("/usr/include/awesome"), "/usr/include/");
        NANO_CHECK_EQUAL(nano::dirname("/usr/include/awesome.txt"), "/usr/include/");
}

NANO_END_MODULE()
