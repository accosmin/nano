#pragma once

#include "arch.h"
#include <string>
#include <iostream>
#include "math/numeric.h"
#include <eigen3/Eigen/Core>

static std::string module_name;
static std::string case_name;
static std::size_t n_cases = 0;
static std::size_t n_checks = 0;
static std::size_t n_failures = 0;

enum class exception_status
{
        none,
        expected,
        unexpected
};

template <typename texception, typename toperator>
static exception_status check_throw(const toperator& op)
{
        try
        {
                op();
                return exception_status::none;
        }
        catch (texception&)
        {
                return exception_status::expected;
        }
        catch (...)
        {
                return exception_status::unexpected;
        }
}

#define NANO_BEGIN_MODULE(name) \
int main(int, char* []) \
{ \
        module_name = #name; \
        n_failures = 0;

#define NANO_CASE(name) \
        ++ n_cases; \
        case_name = #name; \
        std::cout << "running test case [" << module_name << "/" << case_name << "] ..." << std::endl;

#define NANO_END_MODULE() \
        if (n_failures > 0) \
        { \
                std::cout << "  failed with " << n_failures << " errors in " << n_checks \
                          << " check" << (n_checks > 0 ? "s" : "") << "!" << std::endl; \
                exit(EXIT_FAILURE); \
        } \
        else \
        { \
                std::cout << "  no errors detected in " << n_checks \
                          << " check" << (n_checks > 0 ? "s" : "") << "." << std::endl; \
                exit(EXIT_SUCCESS); \
        } \
}

#define NANO_HANDLE_CRITICAL(critical) \
        if (critical) \
        { \
                exit(EXIT_FAILURE); \
        }
#define NANO_HANDLE_FAILURE() \
        ++ n_failures; \
        std::cout << __FILE__ << ":" << __LINE__ << ": [" << module_name << "/" << case_name

#define NANO_EVALUATE(check, critical) \
        ++ n_checks; \
        if (!(check)) \
        { \
                NANO_HANDLE_FAILURE() \
                        << "]: check {" << NANO_STRINGIFY(check) << "} failed!" << std::endl; \
                NANO_HANDLE_CRITICAL(critical) \
        }
#define NANO_CHECK(check) \
        NANO_EVALUATE(check, false)
#define NANO_REQUIRE(check) \
        NANO_EVALUATE(check, true)

#define NANO_THROW(call, exception, critical) \
        ++ n_checks; \
        switch (check_throw<exception>([&] () { (void)(call); })) \
        { \
        case exception_status::none: \
                NANO_HANDLE_FAILURE() \
                        << "]: call {" << NANO_STRINGIFY(call) << "} does not throw!" << std::endl; \
                NANO_HANDLE_CRITICAL(critical) \
        case exception_status::expected: \
                break; \
        case exception_status::unexpected: \
                NANO_HANDLE_FAILURE() \
                        << "]: call {" << NANO_STRINGIFY(call) << "} does not throw {" \
                        << NANO_STRINGIFY(exception) << "}!" << std::endl; \
                NANO_HANDLE_CRITICAL(critical) \
        }
#define NANO_CHECK_THROW(call, exception) \
        NANO_THROW(call, exception, false)
#define NANO_REQUIRE_THROW(call, exception) \
        NANO_THROW(call, exception, true)

#define NANO_NOTHROW(call, critical) \
        ++ n_checks; \
        switch (check_throw<std::exception>([&] () { (void)(call); })) \
        { \
        case exception_stats::none: \
                break; \
        case exception_status::expected: \
        case exception_status::unexpected: \
                NANO_HANDLE_FAILURE() \
                        << "]: call {" << NANO_STRINGIFY(call) << "} throws!" << std::endl; \
                NANO_HANDLE_CRITICAL(critical) \
        }
#define NANO_CHECK_NOTHROW(call) \
        NANO_NOTHROW(call, false)
#define NANO_REQUIRE_NOTHROW(call) \
        NANO_NOTHROW(call, true)

#define NANO_EVALUATE_BINARY_OP(left, right, op, critical) \
        ++ n_checks; \
        if (!((left) op (right))) \
        { \
                NANO_HANDLE_FAILURE() \
                        << "]: check {" << NANO_STRINGIFY(left op right) \
                        << "} failed {" << (left) << " " << NANO_STRINGIFY(op) << " " << (right) << "}!" << std::endl; \
                NANO_HANDLE_CRITICAL(critical) \
        }

#define NANO_EVALUATE_EQUAL(left, right, critical) \
        NANO_EVALUATE_BINARY_OP(left, right, ==, critical)
#define NANO_CHECK_EQUAL(left, right) \
        NANO_EVALUATE_EQUAL(left, right, false)
#define NANO_REQUIRE_EQUAL(left, right) \
        NANO_EVALUATE_EQUAL(left, right, true)

#define NANO_EVALUATE_LESS(left, right, critical) \
        NANO_EVALUATE_BINARY_OP(left, right, <, critical)
#define NANO_CHECK_LESS(left, right) \
        NANO_EVALUATE_LESS(left, right, false)
#define NANO_REQUIRE_LESS(left, right) \
        NANO_EVALUATE_LESS(left, right, true)

#define NANO_EVALUATE_LESS_EQUAL(left, right, critical) \
        NANO_EVALUATE_BINARY_OP(left, right, <=, critical)
#define NANO_CHECK_LESS_EQUAL(left, right) \
        NANO_EVALUATE_LESS_EQUAL(left, right, false)
#define NANO_REQUIRE_LESS_EQUAL(left, right) \
        NANO_EVALUATE_LESS_EQUAL(left, right, true)

#define NANO_EVALUATE_GREATER(left, right, critical) \
        NANO_EVALUATE_BINARY_OP(left, right, >, critical)
#define NANO_CHECK_GREATER(left, right) \
        NANO_EVALUATE_GREATER(left, right, false)
#define NANO_REQUIRE_GREATER(left, right) \
        NANO_EVALUATE_GREATER(left, right, true)

#define NANO_EVALUATE_GREATER_EQUAL(left, right, critical) \
        NANO_EVALUATE_BINARY_OP(left, right, >=, critical)
#define NANO_CHECK_GREATER_EQUAL(left, right) \
        NANO_EVALUATE_GREATER_EQUAL(left, right, false)
#define NANO_REQUIRE_GREATER_EQUAL(left, right) \
        NANO_EVALUATE_GREATER_EQUAL(left, right, true)

#define NANO_EVALUATE_CLOSE(left, right, epsilon, critical) \
        NANO_EVALUATE_LESS(nano::abs((left) - (right)), epsilon, critical)
#define NANO_CHECK_CLOSE(left, right, epsilon) \
        NANO_EVALUATE_CLOSE(left, right, epsilon, false)
#define NANO_REQUIRE_CLOSE(left, right, epsilon) \
        NANO_EVALUATE_CLOSE(left, right, epsilon, true)

#define NANO_EVALUATE_EIGEN_CLOSE(left, right, epsilon, critical) \
        NANO_EVALUATE_LESS((((left) - (right)).array().abs().maxCoeff()), epsilon, critical)
#define NANO_CHECK_EIGEN_CLOSE(left, right, epsilon) \
        NANO_EVALUATE_EIGEN_CLOSE(left, right, epsilon, false)
#define NANO_REQUIRE_EIGEN_CLOSE(left, right, epsilon) \
        NANO_EVALUATE_EIGEN_CLOSE(left, right, epsilon, true)
