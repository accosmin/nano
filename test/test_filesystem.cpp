#include "unit_test.hpp"
#include "text/filesystem.h"

ZOB_BEGIN_MODULE(test_filesystem)

ZOB_CASE(filename)
{
        ZOB_CHECK_EQUAL(text::filename("source"), "source");
        ZOB_CHECK_EQUAL(text::filename("source.out"), "source.out");
        ZOB_CHECK_EQUAL(text::filename("a.out.ext"), "a.out.ext");
        ZOB_CHECK_EQUAL(text::filename("/usr/include/awesome"), "awesome");
        ZOB_CHECK_EQUAL(text::filename("/usr/include/awesome.txt"), "awesome.txt");
}

ZOB_CASE(extension)
{
        ZOB_CHECK_EQUAL(text::extension("source"), "");
        ZOB_CHECK_EQUAL(text::extension("source.out"), "out");
        ZOB_CHECK_EQUAL(text::extension("a.out.ext"), "ext");
        ZOB_CHECK_EQUAL(text::extension("/usr/include/awesome"), "");
        ZOB_CHECK_EQUAL(text::extension("/usr/include/awesome.txt"), "txt");
}

ZOB_CASE(stem)
{
        ZOB_CHECK_EQUAL(text::stem("source"), "source");
        ZOB_CHECK_EQUAL(text::stem("source.out"), "source");
        ZOB_CHECK_EQUAL(text::stem("a.out.ext"), "a.out");
        ZOB_CHECK_EQUAL(text::stem("/usr/include/awesome"), "awesome");
        ZOB_CHECK_EQUAL(text::stem("/usr/include/awesome.txt"), "awesome");
}

ZOB_CASE(dirname)
{
        ZOB_CHECK_EQUAL(text::dirname("source"), "./");
        ZOB_CHECK_EQUAL(text::dirname("source.out"), "./");
        ZOB_CHECK_EQUAL(text::dirname("a.out.ext"), "./");
        ZOB_CHECK_EQUAL(text::dirname("/usr/include/awesome"), "/usr/include/");
        ZOB_CHECK_EQUAL(text::dirname("/usr/include/awesome.txt"), "/usr/include/");
}

ZOB_END_MODULE()
