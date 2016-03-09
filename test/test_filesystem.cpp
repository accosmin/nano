#include "unit_test.hpp"
#include "text/filesystem.h"

ZOB_BEGIN_MODULE(test_filesystem)

ZOB_CASE(filename)
{
        ZOB_CHECK_EQUAL(zob::filename("source"), "source");
        ZOB_CHECK_EQUAL(zob::filename("source.out"), "source.out");
        ZOB_CHECK_EQUAL(zob::filename("a.out.ext"), "a.out.ext");
        ZOB_CHECK_EQUAL(zob::filename("/usr/include/awesome"), "awesome");
        ZOB_CHECK_EQUAL(zob::filename("/usr/include/awesome.txt"), "awesome.txt");
}

ZOB_CASE(extension)
{
        ZOB_CHECK_EQUAL(zob::extension("source"), "");
        ZOB_CHECK_EQUAL(zob::extension("source.out"), "out");
        ZOB_CHECK_EQUAL(zob::extension("a.out.ext"), "ext");
        ZOB_CHECK_EQUAL(zob::extension("/usr/include/awesome"), "");
        ZOB_CHECK_EQUAL(zob::extension("/usr/include/awesome.txt"), "txt");
}

ZOB_CASE(stem)
{
        ZOB_CHECK_EQUAL(zob::stem("source"), "source");
        ZOB_CHECK_EQUAL(zob::stem("source.out"), "source");
        ZOB_CHECK_EQUAL(zob::stem("a.out.ext"), "a.out");
        ZOB_CHECK_EQUAL(zob::stem("/usr/include/awesome"), "awesome");
        ZOB_CHECK_EQUAL(zob::stem("/usr/include/awesome.txt"), "awesome");
}

ZOB_CASE(dirname)
{
        ZOB_CHECK_EQUAL(zob::dirname("source"), "./");
        ZOB_CHECK_EQUAL(zob::dirname("source.out"), "./");
        ZOB_CHECK_EQUAL(zob::dirname("a.out.ext"), "./");
        ZOB_CHECK_EQUAL(zob::dirname("/usr/include/awesome"), "/usr/include/");
        ZOB_CHECK_EQUAL(zob::dirname("/usr/include/awesome.txt"), "/usr/include/");
}

ZOB_END_MODULE()
