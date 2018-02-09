#include "utest.h"
#include "iterator.h"

using namespace nano;

NANO_BEGIN_MODULE(test_iterator)

NANO_CASE(fixed_batch)
{
        auto task = get_tasks().get("synth-affine");
        NANO_REQUIRE(task);
        task->config(json_writer_t().object("isize", 3, "osize", 3, "count", 100).str());
        NANO_CHECK(task->load());

        const auto batch = size_t(23);
        const auto fold = fold_t{0, protocol::train};
        const auto fold_size = task->size(fold);

        iterator_t it(*task, fold, batch);
        for (size_t i = 0; i < 1000; ++ i)
        {
                NANO_CHECK_LESS(it.begin(), it.end());
                NANO_CHECK_LESS_EQUAL(it.end(), fold_size);
                NANO_CHECK_LESS_EQUAL(it.end(), it.begin() + batch);

                const auto end = it.end();
                it.next();

                if (end + batch >= fold_size)
                {
                        NANO_CHECK_EQUAL(it.begin(), size_t(0));
                        NANO_CHECK_EQUAL(it.end(), batch);
                }
                else
                {
                        NANO_CHECK_EQUAL(end, it.begin());
                }
        }
}

NANO_CASE(increasing_batch)
{
        auto task = get_tasks().get("synth-affine");
        NANO_REQUIRE(task);
        task->config(json_writer_t().object("isize", 3, "osize", 3, "count", 100).str());
        NANO_CHECK(task->load());

        const auto batch0 = size_t(3);
        const auto factor = scalar_t(1.05);
        const auto fold = fold_t{0, protocol::train};
        const auto fold_size = task->size(fold);

        iterator_t it(*task, fold, batch0, factor);
        for (size_t i = 0; i < 1000; ++ i)
        {
                NANO_CHECK_LESS(it.begin(), it.end());
                NANO_CHECK_LESS_EQUAL(it.end(), fold_size);
                NANO_CHECK_GREATER_EQUAL(it.end(), it.begin() + batch0);

                const auto begin = it.begin();
                const auto end = it.end();
                it.next();

                if (end + (end - begin) >= fold_size)
                {
                        NANO_CHECK_EQUAL(it.begin(), size_t(0));
                }
        }
}

NANO_CASE(shuffle)
{
        auto task = get_tasks().get("synth-affine");
        NANO_REQUIRE(task);
        task->config(json_writer_t().object("isize", 3, "osize", 3, "count", 102).str());
        NANO_CHECK(task->load());

        for (const auto p : {protocol::train, protocol::valid, protocol::test})
        {
                const auto fold = fold_t{0, p};
                const auto size = task->size(fold);

                std::map<size_t, size_t> iohashes;
                for (size_t i = 0; i < size; ++ i)
                {
                        iohashes[task->ihash(fold, i)] = task->ohash(fold, i);
                }

                for (auto t = 0; t < 8; ++ t)
                {
                        task->shuffle(fold);
                        NANO_REQUIRE_EQUAL(task->size(fold), size);

                        for (size_t i = 0; i < size; ++ i)
                        {
                                const auto ihash = task->ihash(fold, i);
                                const auto ohash = task->ohash(fold, i);
                                const auto it = iohashes.find(ihash);

                                NANO_REQUIRE(it != iohashes.end());
                                NANO_CHECK_EQUAL(it->second, ohash);
                        }
                }
        }
}

NANO_CASE(minibatch)
{
        auto task = get_tasks().get("synth-affine");
        NANO_REQUIRE(task);
        task->config(json_writer_t().object("isize", 3, "osize", 3, "count", 102).str());
        NANO_CHECK(task->load());

        for (const auto p : {protocol::train, protocol::valid, protocol::test})
        {
                const auto fold = fold_t{0, p};
                const auto size = task->size(fold);

                for (size_t count = 1; count < std::min(size_t(8), size); ++ count)
                {
                        const auto minibatch = task->get(fold, size_t(0), count);

                        NANO_CHECK_EQUAL(minibatch.idims(), task->idims());
                        NANO_CHECK_EQUAL(minibatch.odims(), task->odims());
                        NANO_CHECK_EQUAL(minibatch.count(), static_cast<tensor_size_t>(count));
                }
        }
}

NANO_END_MODULE()
