#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_buffer"

#include <boost/test/unit_test.hpp>
#include "math/random.hpp"
#include "cortex/file/buffer.h"
#include "cortex/file/mstream.h"
#include <cstdio>

BOOST_AUTO_TEST_CASE(test_buffer)
{
        using namespace cortex;

        const size_t min_size = 3;
        const size_t max_size = 679 * 1024;

        const auto op_check_buffers = [] (const buffer_t& ref_buffer, const buffer_t& buffer)
        {
                BOOST_REQUIRE_EQUAL(buffer.size(), ref_buffer.size());
                for (size_t i = 0; i < ref_buffer.size(); i ++)
                {
                        BOOST_CHECK_EQUAL(buffer[i], ref_buffer[i]);
                }
        };

        for (size_t size = min_size; size <= max_size; size *= 2)
        {
                // generate reference buffer
                buffer_t ref_buffer = cortex::make_buffer(size);
                BOOST_CHECK_EQUAL(ref_buffer.size(), size);

                math::random_t<char> rng;
                for (auto& value : ref_buffer)
                {
                        value = rng();
                }

                // check buffer saving to file
                const std::string path = "buffer.test";

                BOOST_CHECK(cortex::save_buffer(path, ref_buffer));

                // check buffer loading from file
                {
                        buffer_t buffer;
                        BOOST_CHECK(cortex::load_buffer(path, buffer));

                        op_check_buffers(ref_buffer, buffer);
                }

                // check buffer loading from mstream
                {
                        mstream_t stream(ref_buffer.data(), size);

                        buffer_t buffer;
                        BOOST_CHECK(cortex::load_buffer_from_stream(stream, buffer));

                        op_check_buffers(ref_buffer, buffer);
                }
                {
                        mstream_t stream(ref_buffer.data(), size);

                        buffer_t buffer;
                        BOOST_CHECK(cortex::load_buffer_from_stream(stream, size, buffer));

                        op_check_buffers(ref_buffer, buffer);
                }

                // cleanup
                std::remove(path.c_str());
        }
}
