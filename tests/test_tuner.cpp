#include "utest.h"
#include "tuner.h"
#include "core/epsilon.h"

using namespace nano;

enum class enum_type
{
        type1,
        type2,
        type3
};

namespace nano
{
        template <>
        enum_map_t<enum_type> enum_string<enum_type>()
        {
                return
                {
                        { enum_type::type1,     "type1" },
                        { enum_type::type3,     "type3" }
                };
        }
}

static void get(const json_t& json, scalar_t& x)
{
        x = 100;
        from_json(json, "x", x);
}

static void get(const json_t& json, scalar_t& x, scalar_t& y)
{
        x = y = 100;
        from_json(json, "x", x, "y", y);
}

static void get(const json_t& json, scalar_t& x, scalar_t& y, scalar_t& z)
{
        x = y = z = 100;
        from_json(json, "x", x, "y", y, "z", z);
}

static void get(const json_t& json, scalar_t& x, scalar_t& y, scalar_t& z, enum_type& e)
{
        x = y = z = 100;
        e = enum_type::type2;
        from_json(json, "x", x, "y", y, "z", z, "e", e);
}

static bool is_unique(jsons_t configs)
{
        std::sort(configs.begin(), configs.end());
        return std::unique(configs.begin(), configs.end()) == configs.end();
}

NANO_BEGIN_MODULE(test_tuner)

NANO_CASE(tuner1d)
{
        tuner_t tuner;
        tuner.add_finite("x", -3, +3);

        NANO_CHECK_EQUAL(tuner.n_params(), 1u);
        NANO_CHECK_EQUAL(tuner.n_configs(), 2u);

        const auto configs = tuner.get(100);
        NANO_CHECK(is_unique(configs));
        NANO_CHECK_EQUAL(configs.size(), 2u);

        scalar_t x = 0;
        for (const auto& config : configs)
        {
                get(config, x);

                const auto prodx = (x - 3) * (x + 3);
                NANO_CHECK_CLOSE(prodx, 0, epsilon0<scalar_t>());
        }
}

NANO_CASE(tuner2d)
{
        tuner_t tuner;
        tuner.add_finite("x", -3, +3, +2);
        tuner.add_pow10s("y", -1, +0, +1);

        NANO_CHECK_EQUAL(tuner.n_params(), 2u);
        NANO_CHECK_EQUAL(tuner.n_configs(), 12u);

        const auto configs = tuner.get(10);
        NANO_CHECK(is_unique(configs));
        NANO_CHECK_EQUAL(configs.size(), 10u);

        scalar_t x = 0, y = 0;
        for (const auto& config : configs)
        {
                get(config, x, y);

                const auto prodx = (x - 3) * (x + 3) * (x - 2);
                const auto prody = (y + 1 - 1) * (y + 1 - 3) * (y + 1 - 10) * (y + 1 - 30);
                NANO_CHECK_CLOSE(prodx, 0, epsilon0<scalar_t>());
                NANO_CHECK_CLOSE(prody, 0, epsilon0<scalar_t>());
        }
}

NANO_CASE(tuner3d)
{
        tuner_t tuner;
        tuner.add_finite("x", -3, +3, +2);
        tuner.add_pow10s("y", -1, +0, +1);
        tuner.add_finite("z", 0, 1);

        NANO_CHECK_EQUAL(tuner.n_params(), 3u);
        NANO_CHECK_EQUAL(tuner.n_configs(), 24u);

        const auto configs = tuner.get(20);
        NANO_CHECK(is_unique(configs));
        NANO_CHECK_EQUAL(configs.size(), 20u);

        scalar_t x = 0, y = 0, z = 0;
        for (const auto& config : configs)
        {
                get(config, x, y, z);

                const auto prodx = (x - 3) * (x + 3) * (x - 2);
                const auto prody = (y + 1 - 1) * (y + 1 - 3) * (y + 1 - 10) * (y + 1 - 30);
                const auto prodz = (z - 0) * (z - 1);
                NANO_CHECK_CLOSE(prodx, 0, epsilon0<scalar_t>());
                NANO_CHECK_CLOSE(prody, 0, epsilon0<scalar_t>());
                NANO_CHECK_CLOSE(prodz, 0, epsilon0<scalar_t>());
        }
}

NANO_CASE(tuner3dE)
{
        tuner_t tuner;
        tuner.add_finite("x", -3, +3, +2);
        tuner.add_pow10s("y", -1, +0, +1);
        tuner.add_finite("z", 0, 1);
        tuner.add_enum<enum_type>("e");

        NANO_CHECK_EQUAL(tuner.n_params(), 4u);
        NANO_CHECK_EQUAL(tuner.n_configs(), 48u);

        const auto configs = tuner.get(20);
        NANO_CHECK(is_unique(configs));
        NANO_CHECK_EQUAL(configs.size(), 20u);

        scalar_t x = 0, y = 0, z = 0;
        enum_type e = enum_type::type2;
        for (const auto& config : configs)
        {
                get(config, x, y, z, e);

                const auto prodx = (x - 3) * (x + 3) * (x - 2);
                const auto prody = (y + 1 - 1) * (y + 1 - 3) * (y + 1 - 10) * (y + 1 - 30);
                const auto prodz = (z - 0) * (z - 1);
                NANO_CHECK_CLOSE(prodx, 0, epsilon0<scalar_t>());
                NANO_CHECK_CLOSE(prody, 0, epsilon0<scalar_t>());
                NANO_CHECK_CLOSE(prodz, 0, epsilon0<scalar_t>());
                NANO_CHECK(e == enum_type::type1 || e == enum_type::type3);
        }
}

NANO_END_MODULE()
