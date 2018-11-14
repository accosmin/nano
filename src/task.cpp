#include "tasks/task_mnist.h"
#include "tasks/task_cifar10.h"
#include "tasks/task_cifar100.h"
#include "tasks/task_svhn.h"
#include "tasks/task_iris.h"
#include "tasks/task_wine.h"
#include "tasks/task_affine.h"
#include "tasks/task_peak2d.h"
#include "tasks/task_parity.h"
#include "tasks/task_cal_housing.h"
#include "core/table.h"
#include <mutex>
#include <iostream>

using namespace nano;

task_factory_t& nano::get_tasks()
{
        static task_factory_t manager;

        static std::once_flag flag;
        std::call_once(flag, [] ()
        {
                manager.add<mnist_task_t>("mnist", "MNIST (1x28x28 digit classification)");
                manager.add<fashion_mnist_task_t>("fashion-mnist", "Fashion-MNIST (1x28x28 fashion article classification)");
                manager.add<cifar10_task_t>("cifar10", "CIFAR-10 (3x32x32 object classification)");
                manager.add<cifar100_task_t>("cifar100", "CIFAR-100 (3x32x32 object classification)");
                manager.add<svhn_task_t>("svhn", "SVHN (3x32x32 digit classification in the wild)");
                manager.add<iris_task_t>("iris", "IRIS (iris flower classification)");
                manager.add<wine_task_t>("wine", "WINE (wine classification)");
                manager.add<parity_task_t>("synth-parity", "synthetic: predict the parity bit");
                manager.add<affine_task_t>("synth-affine", "synthetic: predict random noisy affine transformations");
                manager.add<peak2d_task_t>("synth-peak2d", "synthetic: predict random peaks in noisy images");
                manager.add<cal_housing_task_t>("cal-housing", "California median housing value (regression)");
        });

        return manager;
}

template <typename tvalues>
static size_t count_duplicates(const tvalues& values)
{
        size_t count = 0;
        auto it = values.begin();
        while ((it = std::adjacent_find(it, values.end())) != values.end())
        {
                ++ count;
                ++ it;
        }
        return count;
}

template <typename tvalues>
static size_t count_intersects(const tvalues& values1, const tvalues& values2)
{
        tvalues intersection;
        std::set_intersection(
                values1.begin(), values1.end(), values2.begin(), values2.end(),
                std::back_inserter(intersection));
        return intersection.size();
}

template <typename thashes>
static void add_hashes(const task_t& task, const fold_t& fold, thashes& hashes)
{
        const auto size = task.size(fold);
        for (size_t i = 0; i < size; ++ i)
        {
                hashes.push_back(task.ihash(fold, i));
        }
}

void task_t::describe(const string_t& name) const
{
        const auto glcounts = labels();

        std::map<fold_t, std::map<string_t, size_t>> flcounts;
        for (size_t f = 0; f < fsize(); ++ f)
        {
                for (const auto p : {protocol::train, protocol::valid, protocol::test})
                {
                        const auto fold = fold_t{f, p};
                        flcounts[fold] = labels(fold);
                }
        }

        table_t table;
        {
                auto& row = table.append();
                row << colspan(2 + 3 * fsize()) << alignment::center << colfill('=')
                    << (' ' + name + ": " + to_string(idims()) + " -> " + to_string(odims()) + ' ');
        }
        table.delim();
        {
                auto& row = table.append();
                row << "#labels";
                for (size_t f = 0; f < fsize(); ++ f)
                {
                        row << (to_string(f + 1) + ':' + to_string(protocol::train));
                        row << (to_string(f + 1) + ':' + to_string(protocol::valid));
                        row << (to_string(f + 1) + ':' + to_string(protocol::test));
                }
                row << "total";
        }
        table.delim();
        for (const auto& glcount : glcounts)
        {
                auto& row = table.append();
                row << glcount.first;
                for (size_t f = 0; f < fsize(); ++ f)
                {
                        for (const auto p : {protocol::train, protocol::valid, protocol::test})
                        {
                                const auto fold = fold_t{f, p};
                                const auto it = flcounts.find(fold);
                                assert(it != flcounts.end());

                                const auto itx = it->second.find(glcount.first);
                                row << (itx == it->second.end() ? size_t(0) : itx->second);
                        }
                }

                row << glcount.second;
        }
        table.delim();
        {
                auto& row = table.append();
                row << "#samples";
                for (size_t f = 0; f < fsize(); ++ f)
                {
                        row << size({f, protocol::train})
                            << size({f, protocol::valid})
                            << size({f, protocol::test});
                }
                row << size();
        }
        table.delim();
        {
                auto& row = table.append();
                row << "#samples";
                for (size_t f = 0; f < fsize(); ++ f)
                {
                        const auto count =
                                size({f, protocol::train}) +
                                size({f, protocol::valid}) +
                                size({f, protocol::test});
                        row << colspan(3) << alignment::center << count;
                }
                row << colfill(' ') << size();
        }

        std::cout << table;
}

size_t task_t::duplicates(const size_t f) const
{
        assert(f < fsize());

        std::vector<size_t> hashes;

        add_hashes(*this, fold_t{f, protocol::train}, hashes);
        add_hashes(*this, fold_t{f, protocol::valid}, hashes);
        add_hashes(*this, fold_t{f, protocol::test}, hashes);

        return count_duplicates(hashes);
}

size_t task_t::duplicates() const
{
        size_t max_duplicates = 0;
        for (size_t f = 0; f < fsize(); ++ f)
        {
                max_duplicates = std::max(max_duplicates, duplicates(f));
        }
        return max_duplicates;
}

size_t task_t::intersections(const size_t f) const
{
        assert(f < fsize());

        std::vector<size_t> train_hashes;
        std::vector<size_t> valid_hashes;
        std::vector<size_t> test_hashes;

        add_hashes(*this, fold_t{f, protocol::train}, train_hashes);
        add_hashes(*this, fold_t{f, protocol::valid}, valid_hashes);
        add_hashes(*this, fold_t{f, protocol::test}, test_hashes);

        return  std::max(std::max(
                count_intersects(train_hashes, valid_hashes),
                count_intersects(valid_hashes, test_hashes)),
                count_intersects(test_hashes, train_hashes));
}

size_t task_t::intersections() const
{
        size_t max_duplicates = 0;
        for (size_t f = 0; f < fsize(); ++ f)
        {
                max_duplicates = std::max(max_duplicates, intersections(f));
        }
        return max_duplicates;
}

std::map<string_t, size_t> task_t::labels(const fold_t& fold) const
{
        std::map<string_t, size_t> labels;
        for (size_t i = 0, size = this->size(fold); i < size; ++ i)
        {
                labels[label(fold, i)] ++;
        }

        return labels;
}

std::map<string_t, size_t> task_t::labels() const
{
        std::map<string_t, size_t> labels;
        for (size_t f = 0; f < fsize(); ++ f)
        {
                for (const auto p : {protocol::train, protocol::valid, protocol::test})
                {
                        const auto fold = fold_t{f, p};
                        for (size_t i = 0, size = this->size(fold); i < size; ++ i)
                        {
                                labels[label(fold, i)] ++;
                        }
                }
        }

        return labels;
}
