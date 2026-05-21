#include "tasks/mixed_test_classic.hpp"

#include "task_utils.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kMinMaxSegmentsForMixedTest = 10485760;
constexpr int kMaxTableRowsForMixedTest = 2000;
using Real = long double;
constexpr Real kGamma1 = 1.0L;
constexpr Real kGamma2 = 2.0L;
constexpr Real kTheta1 = 3.0L;
constexpr Real kTheta2 = 4.0L;

Real u(Real x, const VariantData& data) {
    if (x < 0.0L || x > 1.0L) {
        throw std::out_of_range("x out of range");
    }

    Real c1 = 0.0L;
    Real c2 = 0.0L;
    Real lambda = 0.0L;
    Real partial = 0.0L;

    if (x < static_cast<Real>(data.xi)) {
        c1 = 2.137730248054328L;
        c2 = 2.597069887476066L;
        lambda = 1.0L / std::sqrt(3.0L);
        partial = -2.0L;
    } else {
        c1 = 0.736243285201879L;
        c2 = 1.191698220955875L;
        lambda = std::sqrt(10.0L / (9.0L * std::exp(1.0L / 3.0L)));
        partial = 0.9L;
    }

    return c1 * std::exp(lambda * x) + c2 * std::exp(-lambda * x) + partial;
}

struct GridSolution {
    int n = 0;
    std::vector<Real> values;
};

struct AccuracyCheck {
    Real epsilon = 0.0L;
    Real x = 0.0L;
    int index = 0;
};

Real integrateK(Real left, Real right, Real xi) {

    const Real k1_star = 1.0L;
    const Real k2_star = std::exp(1.0L / 3.0L);
    const Real inv_k1 = 1.0L / k1_star;  // = 1
    const Real inv_k2 = 1.0L / k2_star;  // = e^{-1/3}
    
    if (right <= left) return 0.0L;
    if (right <= xi) return inv_k1 * (right - left);
    if (left >= xi) return inv_k2 * (right - left);
    return inv_k1 * (xi - left) + inv_k2 * (right - xi);
}

Real integrateQ(Real left, Real right, Real xi) {
    const Real q1 = 1.0L / 3.0L;
    const Real q2 = 10.0L / 9.0L;
    
    if (right <= left) return 0.0L;
    if (right <= xi) return q1 * (right - left);
    if (left >= xi) return q2 * (right - left);
    return q1 * (xi - left) + q2 * (right - xi);
}

Real integrateF(Real left, Real right, Real xi) {
    const Real f1 = -2.0L / 3.0L;
    const Real f2 = 1.0L;
    
    if (right <= left) return 0.0L;
    if (right <= xi) return f1 * (right - left);
    if (left >= xi) return f2 * (right - left);
    return f1 * (xi - left) + f2 * (right - xi);
}

Real coefficientA(int i, Real h, Real xi) {
    const Real left = static_cast<Real>(i - 1) * h;
    const Real right = static_cast<Real>(i) * h;
    const Real integral = integrateK(left, right, xi);
    if (integral <= 0.0L) {
        throw std::runtime_error("Invalid integral for coefficient a_i");
    }
    return h / integral;
}

Real coefficientD(int i, int n, Real h, Real xi) {
    const Real x = static_cast<Real>(i) * h;
    const Real left = std::max(0.0L, x - 0.5L * h);
    const Real right = std::min(1.0L, x + 0.5L * h);
    const Real width = right - left;
    return integrateQ(left, right, xi) / width;
}

Real coefficientPhi(int i, int n, Real h, Real xi) {
    const Real x = static_cast<Real>(i) * h;
    const Real left = std::max(0.0L, x - 0.5L * h);
    const Real right = std::min(1.0L, x + 0.5L * h);
    const Real width = right - left;
    return integrateF(left, right, xi) / width;
}

std::vector<Real> solveTridiagonalLongDouble(
    const std::vector<Real>& lower,
    const std::vector<Real>& diagonal,
    const std::vector<Real>& upper,
    const std::vector<Real>& rhs) {

    const size_t n = diagonal.size();
    if (rhs.size() != n || lower.size() + 1 < n || upper.size() + 1 < n) {
        throw std::runtime_error("Invalid tridiagonal system size");
    }

    std::vector<Real> cPrime(n, 0.0L);
    std::vector<Real> dPrime(n, 0.0L);

    if (std::abs(diagonal[0]) < 1e-30L) {
        throw std::runtime_error("Degenerate tridiagonal matrix");
    }

    cPrime[0] = n > 1 ? upper[0] / diagonal[0] : 0.0L;
    dPrime[0] = rhs[0] / diagonal[0];

    for (size_t i = 1; i < n; ++i) {
        const Real denom = diagonal[i] - lower[i - 1] * cPrime[i - 1];
        if (std::abs(denom) < 1e-30L) {
            throw std::runtime_error("Degenerate tridiagonal matrix");
        }
        cPrime[i] = (i + 1 < n) ? upper[i] / denom : 0.0L;
        dPrime[i] = (rhs[i] - lower[i - 1] * dPrime[i - 1]) / denom;
    }

    std::vector<Real> x(n, 0.0L);
    x[n - 1] = dPrime[n - 1];
    for (size_t i = n - 1; i-- > 0;) {
        x[i] = dPrime[i] - cPrime[i] * x[i + 1];
    }
    return x;
}

GridSolution solveForN(int n, const VariantData& variant) {
    if (n < 1) {
        throw std::runtime_error("n must be positive");
    }

    const int size = n + 1;
    const Real h = 1.0L / static_cast<Real>(n);
    const Real xi = static_cast<Real>(variant.xi);

    std::vector<Real> lower(static_cast<size_t>(size - 1), 0.0L);
    std::vector<Real> diagonal(static_cast<size_t>(size), 0.0L);
    std::vector<Real> upper(static_cast<size_t>(size - 1), 0.0L);
    std::vector<Real> rhs(static_cast<size_t>(size), 0.0L);

    const Real aLeftBoundary = coefficientA(1, h, xi);
    diagonal[0] = aLeftBoundary / h + kGamma1;
    upper[0] = -aLeftBoundary / h;
    rhs[0] = kGamma1 * kTheta1;

    for (int i = 1; i <= n - 1; ++i) {
        const Real aLeft = coefficientA(i, h, xi);
        const Real aRight = coefficientA(i + 1, h, xi);
        const Real d = coefficientD(i, n, h, xi);
        const Real phi = coefficientPhi(i, n, h, xi);

        lower[static_cast<size_t>(i - 1)] = -aLeft / h;
        diagonal[static_cast<size_t>(i)] = (aLeft + aRight) / h + d * h;
        upper[static_cast<size_t>(i)] = -aRight / h;
        rhs[static_cast<size_t>(i)] = phi * h;
    }

    const Real aRightBoundary = coefficientA(n, h, xi);
    lower[static_cast<size_t>(n - 1)] = -aRightBoundary / h;
    diagonal[static_cast<size_t>(n)] = aRightBoundary / h + kGamma2;
    rhs[static_cast<size_t>(n)] = kGamma2 * kTheta2;

    return GridSolution{n, solveTridiagonalLongDouble(lower, diagonal, upper, rhs)};
}

AccuracyCheck compareWithAnalytic(const GridSolution& comp, const VariantData& variant) {
    AccuracyCheck check;
    for (int i = 0; i <= comp.n; ++i) {
        const Real x = static_cast<Real>(i) / static_cast<Real>(comp.n);
        const Real difference = comp.values[static_cast<size_t>(i)] - u(x, variant);
        const Real absDifference = std::abs(difference);
        if (absDifference > check.epsilon) {
            check.epsilon = absDifference;
            check.index = i;
            check.x = x;
        }
    }
    return check;
}

struct RefinedSolution {
    GridSolution computed;
    AccuracyCheck check;
    bool meetsTolerance = false;
    bool stoppedOnErrorGrowth = false;
};

RefinedSolution solveWithRefinement(const InputData& input, const VariantData& variant) {
    int n = std::max(1, input.segments);
    
    if (n > input.maxSegments) {
        n = input.maxSegments;
    }

    GridSolution comp = solveForN(n, variant);
    AccuracyCheck check = compareWithAnalytic(comp, variant);
    const bool meetsTolerance = check.epsilon <= static_cast<Real>(input.tolerance);

    return RefinedSolution{std::move(comp), check, meetsTolerance, false};
}

std::string formatScientific(Real value) {
    std::ostringstream out;
    out << std::scientific << std::setprecision(6) << static_cast<double>(value);
    return out.str();
}

void fillRows(TaskResult& task, const RefinedSolution& solution, const VariantData& variant, int tableStride) {
    const int stride = std::max(1, tableStride);
    const int n = solution.computed.n;

    for (int i = 0; i <= n; i += stride) {
        const Real x = static_cast<Real>(i) / static_cast<Real>(n);
        const Real v = solution.computed.values[static_cast<size_t>(i)];
        const Real exact = u(x, variant);
        task.rows.push_back(TableRow{
            i,
            static_cast<double>(x),
            static_cast<double>(exact),
            static_cast<double>(v),
            0.0,
            static_cast<double>(exact - v)
        });
    }

    if (task.rows.empty() || task.rows.back().index != n) {
        const Real v = solution.computed.values[static_cast<size_t>(n)];
        const Real exact = u(1.0L, variant);
        task.rows.push_back(TableRow{n, 1.0, static_cast<double>(exact), static_cast<double>(v), 0.0, static_cast<double>(exact - v)});
    }
}

} //namespace

TaskResult runMixedTestClassicTask(const InputData& input, const VariantData& variant) {
    TaskResult result = makeTaskStub(
        "mixed-test-classic",
        "Смешанная краевая тестовая задача, классическая аппроксимация ГУ",
        "3. Смешанная тест.",
        "w(0)=-gamma1*(u(0)-theta1), w(1)=-gamma2*(theta2-u(1))",
        "Классическая аппроксимация граничных условий",
        "Романова Василиса",
        makeTestTaskColumns());

    const RefinedSolution solution = solveWithRefinement(input, variant);
    const int outputStride = std::max({
        1,
        input.tableStride,
        solution.computed.n / kMaxTableRowsForMixedTest
    });
    fillRows(result, solution, variant, outputStride);

    std::ostringstream note;
    note << "Для тестовой смешанной задачи использована равномерная сетка с n = "
         << solution.computed.n << ".\n"
         << "Граничные условия 3-го рода: w(0)=-gamma1*(u(0)-theta1), w(1)=-gamma2*(theta2-u(1)).\n"
         << "gamma1 = " << static_cast<double>(kGamma1) << ", gamma2 = " << static_cast<double>(kGamma2)
         << ", theta1 = " << static_cast<double>(kTheta1) << ", theta2 = " << static_cast<double>(kTheta2) << ".\n"
         << "Классическая аппроксимация ГУ: k(0)(v_1-v_0)/h = gamma1*(v_0-theta1), "
         << "k(1)(v_n-v_{n-1})/h = gamma2*(theta2-v_n).\n"
         << "Коэффициенты тестовой задачи k*, q*, f* заданы как предел соответствующей функции при x -> xi слева или справа и равны:\n"
         << "k1* = 1; k2* = exp(1/3);\n"
         << "q1* = 1/3; q2* = 10/9;\n"
         << "f1* = -2/3; f2* = 1.\n"
         << "Достигнутая точность epsilon_1 = max|u(x_i)-v(x_i)| = "
         << formatScientific(solution.check.epsilon) << ".\n"
         << "Максимальная разность наблюдается в узле i = " << solution.check.index
         << ", x = " << std::setprecision(10) << solution.check.x << ".\n";

    result.status = "done";
    result.note = note.str();
    return result;
}
