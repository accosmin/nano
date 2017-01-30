#include "utest.h"
#include "io/istream.h"
#include "io/ibstream.h"
#include "io/obstream.h"
#include "math/random.h"
#include "io/istream_mem.h"
#include "io/istream_std.h"
#include <cstdio>
#include <fstream>

nano::buffer_t load_buffer(nano::istream_t& stream, const std::size_t buff_size)
{
        nano::buffer_t buff, data;
        buff.resize(buff_size);
        while (stream)
        {
                stream.read(buff.data(), static_cast<std::streamsize>(buff_size));
                data.insert(data.end(), buff.data(), buff.data() + stream.gcount());
        }
        return data;
}

NANO_BEGIN_MODULE(test_io)

NANO_CASE(istream)
{
        using namespace nano;

        const size_t min_size = 3;
        const size_t max_size = 679 * 1024;

        nano::random_t<char> rng_value;
        nano::random_t<std::streamsize> rng_skip(1, 1024);

        for (size_t size = min_size; size <= max_size; size *= 2)
        {
                // generate reference buffer
                buffer_t ref_buffer = nano::make_buffer(size);
                NANO_CHECK_EQUAL(ref_buffer.size(), size);

                for (auto& value : ref_buffer)
                {
                        value = rng_value();
                }

                // check saving to file
                const std::string path = "mstream.test";

                NANO_CHECK(nano::save_buffer(path, ref_buffer));

                // check loading from file
                {
                        buffer_t buffer;
                        NANO_CHECK(nano::load_buffer(path, buffer));
                        NANO_REQUIRE_EQUAL(buffer.size(), ref_buffer.size());
                        NANO_CHECK(std::equal(buffer.begin(), buffer.end(), ref_buffer.begin()));
                }

                // check loading from memory stream (by block)
                {
                        mem_istream_t stream(ref_buffer.data(), size);

                        NANO_CHECK_EQUAL(stream.tellg(), std::streamsize(0));
                        const buffer_t buffer = load_buffer(stream, size % 43);
                        NANO_REQUIRE_EQUAL(buffer.size(), ref_buffer.size());
                        NANO_CHECK(std::equal(buffer.begin(), buffer.end(), ref_buffer.begin()));
                        NANO_CHECK_EQUAL(stream.tellg(), static_cast<std::streamsize>(size));
                }

                // check loading from std::istream wrapper (by block)
                {
                        std::ifstream istream(path.c_str(), std::ios::binary | std::ios::in);
                        NANO_REQUIRE(istream.is_open());
                        std_istream_t stream(istream);

                        NANO_CHECK_EQUAL(stream.tellg(), std::streamsize(0));
                        const buffer_t buffer = load_buffer(stream, size % 17);
                        NANO_REQUIRE_EQUAL(buffer.size(), ref_buffer.size());
                        NANO_CHECK(std::equal(buffer.begin(), buffer.end(), ref_buffer.begin()));
                        NANO_CHECK_EQUAL(stream.tellg(), static_cast<std::streamsize>(size));
                }

                // check random skip ranges
                {
                        mem_istream_t stream(ref_buffer.data(), size);

                        NANO_CHECK_EQUAL(stream.tellg(), std::streamsize(0));
                        std::streamsize remaining = static_cast<std::streamsize>(size);
                        while (stream)
                        {
                                NANO_REQUIRE_GREATER(remaining, 0);
                                const std::streamsize skip_size = std::min(remaining, rng_skip());
                                NANO_CHECK(stream.skip(skip_size));
                                remaining -= skip_size;
                        }
                        NANO_CHECK_EQUAL(stream.tellg(), static_cast<std::streamsize>(size));
                }

                // cleanup
                std::remove(path.c_str());
        }
}

NANO_CASE(bstream)
{
        struct pod_t
        {
                int i;
                float f;
                double d;
        };

        const double var_double = -1.45;
        const std::string var_string = "string to write";
        const float var_float = 45.7f;
        const int var_int = 393440;
        const std::size_t var_size_t = 323203023;
        const pod_t var_pod = { 45, 23.6f, -4.389384934 };
        const std::vector<short> var_shorts = { 13, -26, 39, -52 };

        const std::string path = "bstream.test";

        // check writing
        {
                std::ofstream os(path.c_str(), std::ios::binary | std::ios::trunc);

                nano::obstream_t ob(os);

                ob.write(var_double);
                ob.write(var_string);
                ob.write(var_float);
                ob.write(var_int);
                ob.write(var_size_t);
                ob.write(var_pod);
                ob.write(var_shorts);

                NANO_REQUIRE(os.good());
        }

        // check reading
        {
                std::ifstream is(path.c_str(), std::ios::binary);

                nano::ibstream_t ib(is);

                double var_double_ex;
                std::string var_string_ex;
                float var_float_ex;
                int var_int_ex;
                std::size_t var_size_t_ex;
                pod_t var_pod_ex;
                std::vector<short> var_shorts_ex;

                ib.read(var_double_ex);
                ib.read(var_string_ex);
                ib.read(var_float_ex);
                ib.read(var_int_ex);
                ib.read(var_size_t_ex);
                ib.read(var_pod_ex);
                ib.read(var_shorts_ex);

                NANO_CHECK_EQUAL(var_double, var_double_ex);
                NANO_CHECK_EQUAL(var_string, var_string_ex);
                NANO_CHECK_EQUAL(var_float, var_float_ex);
                NANO_CHECK_EQUAL(var_int, var_int_ex);
                NANO_CHECK_EQUAL(var_size_t, var_size_t_ex);
                NANO_CHECK_EQUAL(var_pod.d, var_pod_ex.d);
                NANO_CHECK_EQUAL(var_pod.f, var_pod_ex.f);
                NANO_CHECK_EQUAL(var_pod.i, var_pod_ex.i);
                NANO_REQUIRE_EQUAL(var_shorts.size(), var_shorts_ex.size());
                NANO_CHECK(std::equal(var_shorts.begin(), var_shorts.end(), var_shorts_ex.begin()));

                NANO_CHECK(is);
        }

        // cleanup
        std::remove(path.c_str());
}

NANO_END_MODULE()
