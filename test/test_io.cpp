#include "unit_test.hpp"
#include "io/buffer.h"
#include "io/imstream.h"
#include "io/ibstream.h"
#include "io/obstream.h"
#include "math/random.hpp"
#include <cstdio>
#include <fstream>

NANOCV_BEGIN_MODULE(test_io)

NANOCV_CASE(mstream)
{
        using namespace io;

        const size_t min_size = 3;
        const size_t max_size = 679 * 1024;

        const auto op_check_buffers = [] (const buffer_t& ref_buffer, const buffer_t& buffer)
        {
                NANOCV_REQUIRE_EQUAL(buffer.size(), ref_buffer.size());
                NANOCV_CHECK(std::equal(buffer.begin(), buffer.end(), ref_buffer.begin()));
        };

        for (size_t size = min_size; size <= max_size; size *= 2)
        {
                // generate reference buffer
                buffer_t ref_buffer = io::make_buffer(size);
                NANOCV_CHECK_EQUAL(ref_buffer.size(), size);

                math::random_t<char> rng;
                for (auto& value : ref_buffer)
                {
                        value = rng();
                }

                // check buffer saving to file
                const std::string path = "mstream.test";

                NANOCV_CHECK(io::save_buffer(path, ref_buffer));

                // check buffer loading from file
                {
                        buffer_t buffer;
                        NANOCV_CHECK(io::load_buffer(path, buffer));

                        op_check_buffers(ref_buffer, buffer);
                }

                // check buffer loading from mstream (by block)
                {
                        imstream_t stream(ref_buffer.data(), size);

                        NANOCV_CHECK_EQUAL(stream.tellg(), std::streamsize(0));
                        NANOCV_CHECK_EQUAL(stream.size(), static_cast<std::streamsize>(size));

                        buffer_t buffer;
                        NANOCV_CHECK(io::load_buffer_from_stream(stream, buffer));

                        op_check_buffers(ref_buffer, buffer);
                }

                // check buffer loading from mstream (one character at a time)
                {
                        imstream_t stream(ref_buffer.data(), size);

                        NANOCV_CHECK_EQUAL(stream.tellg(), std::streamsize(0));
                        NANOCV_CHECK_EQUAL(stream.size(), static_cast<std::streamsize>(size));

                        buffer_t buffer;
                        char ch;
                        while (stream.read(&ch, 1))
                        {
                                buffer.push_back(ch);
                        }

                        op_check_buffers(ref_buffer, buffer);
                }

                // cleanup
                std::remove(path.c_str());
        }
}

NANOCV_CASE(bstream)
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

                io::obstream_t ob(os);

                ob.write(var_double);
                ob.write(var_string);
                ob.write(var_float);
                ob.write(var_int);
                ob.write(var_size_t);
                ob.write(var_pod);
                ob.write(var_shorts);

                NANOCV_REQUIRE(os.good());
        }

        // check reading
        {
                std::ifstream is(path.c_str(), std::ios::binary);

                io::ibstream_t ib(is);

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

                NANOCV_CHECK_EQUAL(var_double, var_double_ex);
                NANOCV_CHECK_EQUAL(var_string, var_string_ex);
                NANOCV_CHECK_EQUAL(var_float, var_float_ex);
                NANOCV_CHECK_EQUAL(var_int, var_int_ex);
                NANOCV_CHECK_EQUAL(var_size_t, var_size_t_ex);
                NANOCV_CHECK_EQUAL(var_pod.d, var_pod_ex.d);
                NANOCV_CHECK_EQUAL(var_pod.f, var_pod_ex.f);
                NANOCV_CHECK_EQUAL(var_pod.i, var_pod_ex.i);
                NANOCV_REQUIRE_EQUAL(var_shorts.size(), var_shorts_ex.size());
                NANOCV_CHECK(std::equal(var_shorts.begin(), var_shorts.end(), var_shorts_ex.begin()));

                NANOCV_CHECK(is);
        }

        // cleanup
        std::remove(path.c_str());
}

NANOCV_END_MODULE()
