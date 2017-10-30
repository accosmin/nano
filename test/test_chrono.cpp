#include "utest.h"
#include "chrono/probe.h"

NANO_BEGIN_MODULE(test_chrono)

NANO_CASE(gflops)
{
        NANO_CHECK_EQUAL(nano::gflops(42, nano::seconds_t(1)), 0);
        NANO_CHECK_EQUAL(nano::gflops(42, nano::milliseconds_t(1)), 0);
        NANO_CHECK_EQUAL(nano::gflops(42, nano::microseconds_t(1)), 0);
        NANO_CHECK_EQUAL(nano::gflops(42, nano::nanoseconds_t(100)), 0);
        NANO_CHECK_EQUAL(nano::gflops(42, nano::nanoseconds_t(10)), 4);
        NANO_CHECK_EQUAL(nano::gflops(42, nano::nanoseconds_t(1)), 42);
        NANO_CHECK_EQUAL(nano::gflops(42, nano::picoseconds_t(100)), 420);
        NANO_CHECK_EQUAL(nano::gflops(42, nano::picoseconds_t(10)), 4200);
        NANO_CHECK_EQUAL(nano::gflops(42, nano::picoseconds_t(1)), 42000);
}

NANO_CASE(probe)
{
        const auto basename = "base";
        const auto fullname = "full";
        const auto flops = 256;

        nano::probe_t probe(basename, fullname, flops);

        NANO_CHECK_EQUAL(probe.basename(), basename);
        NANO_CHECK_EQUAL(probe.fullname(), fullname);
        NANO_CHECK_EQUAL(probe.flops(), flops);
        NANO_CHECK(!probe);

        probe.measure([] () {});
        probe.measure([] () {});
        probe.measure([] () {});
        probe.measure([] () {});

        NANO_CHECK_EQUAL(probe.flops(), flops);
        NANO_CHECK_EQUAL(probe.gflops(), nano::gflops(flops, nano::nanoseconds_t(probe.timings().min())));
}

NANO_END_MODULE()
