#include "unit_test.hpp"
#include "text/filesystem.h"

NANOCV_BEGIN_MODULE(test_filesystem)

NANOCV_CASE(filename)
{
        NANOCV_CHECK_EQUAL(text::filename("source"), "source");
        NANOCV_CHECK_EQUAL(text::filename("source.out"), "source.out");
        NANOCV_CHECK_EQUAL(text::filename("a.out.ext"), "a.out.ext");
        NANOCV_CHECK_EQUAL(text::filename("/usr/include/awesome"), "awesome");
        NANOCV_CHECK_EQUAL(text::filename("/usr/include/awesome.txt"), "awesome.txt");
}

NANOCV_CASE(extension)
{
        NANOCV_CHECK_EQUAL(text::extension("source"), "");
        NANOCV_CHECK_EQUAL(text::extension("source.out"), "out");
        NANOCV_CHECK_EQUAL(text::extension("a.out.ext"), "ext");
        NANOCV_CHECK_EQUAL(text::extension("/usr/include/awesome"), "");
        NANOCV_CHECK_EQUAL(text::extension("/usr/include/awesome.txt"), "txt");
}

NANOCV_CASE(stem)
{
        NANOCV_CHECK_EQUAL(text::stem("source"), "source");
        NANOCV_CHECK_EQUAL(text::stem("source.out"), "source");
        NANOCV_CHECK_EQUAL(text::stem("a.out.ext"), "a.out");
        NANOCV_CHECK_EQUAL(text::stem("/usr/include/awesome"), "awesome");
        NANOCV_CHECK_EQUAL(text::stem("/usr/include/awesome.txt"), "awesome");
}

NANOCV_CASE(dirname)
{
        NANOCV_CHECK_EQUAL(text::dirname("source"), "./");
        NANOCV_CHECK_EQUAL(text::dirname("source.out"), "./");
        NANOCV_CHECK_EQUAL(text::dirname("a.out.ext"), "./");
        NANOCV_CHECK_EQUAL(text::dirname("/usr/include/awesome"), "/usr/include/");
        NANOCV_CHECK_EQUAL(text::dirname("/usr/include/awesome.txt"), "/usr/include/");
}

NANOCV_END_MODULE()
