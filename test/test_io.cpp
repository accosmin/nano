#include "utest.h"
#include "tensor.h"
#include "io/istream.h"
#include "io/ibstream.h"
#include "io/obstream.h"
#include "math/random.h"
#include "math/epsilon.h"
#include "io/istream_mem.h"
#include "io/istream_std.h"
#include "tensor/numeric.h"
#include <cstdio>
#include <fstream>

using namespace nano;

buffer_t load_buffer(istream_t& stream, const std::size_t buff_size)
{
        buffer_t buff, data;
        buff.resize(buff_size);
        while (stream)
        {
                stream.read(buff.data(), static_cast<std::streamsize>(buff_size));
                data.insert(data.end(), buff.data(), buff.data() + stream.gcount());
        }
        return data;
}

NANO_BEGIN_MODULE(test_io)

NANO_CASE(string)
{
        const std::string path = "string.test";
        const std::string ref_str = "secret sauce 42";

        NANO_CHECK(save_string(path, ref_str));

        std::string str = "testing";

        NANO_CHECK(load_string(path, str));
        NANO_CHECK_EQUAL(str, ref_str);

        NANO_CHECK(load_string(path, str));
        NANO_CHECK_EQUAL(str, ref_str);

        // cleanup
        std::remove(path.c_str());
}

NANO_CASE(istream)
{
        const size_t min_size = 3;
        const size_t max_size = 679 * 1024;

        random_t<char> rng_value;
        random_t<std::streamsize> rng_skip(1, 1024);

        for (size_t size = min_size; size <= max_size; size *= 2)
        {
                // generate reference buffer
                buffer_t ref_buffer = make_buffer(size);
                NANO_CHECK_EQUAL(ref_buffer.size(), size);

                for (auto& value : ref_buffer)
                {
                        value = rng_value();
                }

                // check saving to file
                const std::string path = "mstream.test";

                NANO_CHECK(save_buffer(path, ref_buffer));

                // check loading from file
                {
                        buffer_t buffer;
                        NANO_CHECK(load_buffer(path, buffer));
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
                        auto remaining = static_cast<std::streamsize>(size);
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
        const auto var_struct = pod_t{ 45, 23.6f, -4.389384934 };

        auto var_vector = vector_t(13);
        auto var_matrix = matrix_t(17, 5);
        auto var_tensor = tensor3d_t(3, 4, 5);

        auto rng = make_rng<scalar_t>(-1, +1);
        nano::set_random(rng, var_vector, var_matrix, var_tensor);

        const std::string path = "bstream.test";

        // check writing
        {
                obstream_t ob(path);

                NANO_CHECK(ob.write(var_double));
                NANO_CHECK(ob.write(var_string));
                NANO_CHECK(ob.write(var_float));
                NANO_CHECK(ob.write(var_int));
                NANO_CHECK(ob.write(var_size_t));
                NANO_CHECK(ob.write(var_struct));
                NANO_CHECK(ob.write_vector(var_vector));
                NANO_CHECK(ob.write_matrix(var_matrix));
                NANO_CHECK(ob.write_tensor(var_tensor));
        }

        // check reading
        {
                ibstream_t ib(path);

                double var_double_ex;
                std::string var_string_ex;
                float var_float_ex;
                int var_int_ex;
                std::size_t var_size_t_ex;
                pod_t var_struct_ex;
                vector_t var_vector_ex;
                matrix_t var_matrix_ex;
                tensor3d_t var_tensor_ex;

                NANO_CHECK(ib.read(var_double_ex));
                NANO_CHECK(ib.read(var_string_ex));
                NANO_CHECK(ib.read(var_float_ex));
                NANO_CHECK(ib.read(var_int_ex));
                NANO_CHECK(ib.read(var_size_t_ex));
                NANO_CHECK(ib.read(var_struct_ex));
                NANO_CHECK(ib.read_vector(var_vector_ex));
                NANO_CHECK(ib.read_matrix(var_matrix_ex));
                NANO_CHECK(ib.read_tensor(var_tensor_ex));

                NANO_CHECK_EQUAL(var_double, var_double_ex);
                NANO_CHECK_EQUAL(var_string, var_string_ex);
                NANO_CHECK_EQUAL(var_float, var_float_ex);
                NANO_CHECK_EQUAL(var_int, var_int_ex);
                NANO_CHECK_EQUAL(var_size_t, var_size_t_ex);
                NANO_CHECK_EQUAL(var_struct.d, var_struct_ex.d);
                NANO_CHECK_EQUAL(var_struct.f, var_struct_ex.f);
                NANO_CHECK_EQUAL(var_struct.i, var_struct_ex.i);

                NANO_REQUIRE_EQUAL(var_vector.size(), var_vector_ex.size());
                NANO_REQUIRE_EQUAL(var_matrix.rows(), var_matrix_ex.rows());
                NANO_REQUIRE_EQUAL(var_matrix.cols(), var_matrix_ex.cols());
                NANO_REQUIRE_EQUAL(var_tensor.dims(), var_tensor_ex.dims());

                NANO_CHECK_EIGEN_CLOSE(var_vector, var_vector_ex, epsilon0<scalar_t>());
                NANO_CHECK_EIGEN_CLOSE(var_matrix, var_matrix_ex, epsilon0<scalar_t>());
                NANO_CHECK_EIGEN_CLOSE(var_tensor.vector(), var_tensor_ex.vector(), epsilon0<scalar_t>());
        }

        // cleanup
        std::remove(path.c_str());
}

NANO_END_MODULE()
