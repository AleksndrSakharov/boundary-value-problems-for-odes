#include "tasks/first_dirichlet_test.hpp"

#include "solver.hpp"
#include "task_utils.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <vector>

namespace {

constexpr double kLeft = 1.0;
const double kRight = std::exp(1.0 / 3.0);
constexpr double qLeft = 1.0 / 3.0;
constexpr double qRight = 10.0 / 9.0;
constexpr double fLeft = -2.0 / 3.0;
constexpr double fRight = 1.0;

double integratePiecewiseConst(double left, double right, double xi, double leftValue, double rightValue) {
    if (right <= left) {
        return 0.0;
    }
    if (right <= xi) {
        return leftValue * (right - left);
    }
    if (left >= xi) {
        return rightValue * (right - left);
    }
    return leftValue * (xi - left) + rightValue * (right - xi);
}

double integrateInvK(double left, double right, double xi) {
    return integratePiecewiseConst(left, right, xi, 1.0 / kLeft, 1.0 / kRight);
}

double integrateQ(double left, double right, double xi) {
    return integratePiecewiseConst(left, right, xi, qLeft, qRight);
}

double integrateF(double left, double right, double xi) {
    return integratePiecewiseConst(left, right, xi, fLeft, fRight);
}

struct AnalyticSolution {
    double xi = 0.0;
    double mu1 = 0.0;
    double mu2 = 0.0;

    double lambda1 = 0.0;
    double lambda2 = 0.0;
    double p1 = 0.0;
    double p2 = 0.0;

    double A1 = 0.0;
    double B1 = 0.0;
    double A2 = 0.0;
    double B2 = 0.0;

    explicit AnalyticSolution(double xiIn, double mu1In, double mu2In)
        : xi(xiIn), mu1(mu1In), mu2(mu2In) {
        lambda1 = std::sqrt(qLeft / kLeft);
        lambda2 = std::sqrt(qRight / kRight);
        p1 = fLeft / qLeft;
        p2 = fRight / qRight;

        const double e1 = std::exp(lambda1 * xi);
        const double em1 = std::exp(-lambda1 * xi);
        const double e2 = std::exp(lambda2 * xi);
        const double em2 = std::exp(-lambda2 * xi);

        double system[4][5] = {
            {1.0, 1.0, 0.0, 0.0, mu1 - p1},
            {0.0, 0.0, std::exp(lambda2), std::exp(-lambda2), mu2 - p2},
            {e1, em1, -e2, -em2, p2 - p1},
            {kLeft * lambda1 * e1, -kLeft * lambda1 * em1, -kRight * lambda2 * e2, kRight * lambda2 * em2, 0.0}
        };

        for (int col = 0; col < 4; ++col) {
            int pivot = col;
            double maxVal = std::abs(system[col][col]);
            for (int row = col + 1; row < 4; ++row) {
                double val = std::abs(system[row][col]);
                if (val > maxVal) {
                    maxVal = val;
                    pivot = row;
                }
            }
            if (pivot != col) {
                for (int k = col; k < 5; ++k) {
                    std::swap(system[col][k], system[pivot][k]);
                }
            }

            const double div = system[col][col];
            for (int k = col; k < 5; ++k) {
                system[col][k] /= div;
            }

            for (int row = 0; row < 4; ++row) {
                if (row == col) {
                    continue;
                }
                const double factor = system[row][col];
                for (int k = col; k < 5; ++k) {
                    system[row][k] -= factor * system[col][k];
                }
            }
        }

        A1 = system[0][4];
        B1 = system[1][4];
        A2 = system[2][4];
        B2 = system[3][4];
    }

    double value(double x) const {
        if (x <= xi) {
            return A1 * std::exp(lambda1 * x) + B1 * std::exp(-lambda1 * x) + p1;
        }
        return A2 * std::exp(lambda2 * x) + B2 * std::exp(-lambda2 * x) + p2;
    }
};

std::vector<double> solveGrid(int n, double xi, double mu1, double mu2) {
    const double h = 1.0 / static_cast<double>(n);
    const int interior = n - 1;
    std::vector<double> lower(std::max(0, interior - 1), 0.0);
    std::vector<double> diagonal(interior, 0.0);
    std::vector<double> upper(std::max(0, interior - 1), 0.0);
    std::vector<double> rhs(interior, 0.0);

    for (int i = 1; i <= interior; ++i) {
        const double x = i * h;

        const double aLeft = h / integrateInvK(x - h, x, xi);
        const double aRight = h / integrateInvK(x, x + h, xi);
        const double d = integrateQ(x - 0.5 * h, x + 0.5 * h, xi) / h;
        const double phi = integrateF(x - 0.5 * h, x + 0.5 * h, xi) / h;

        const double A = aLeft / h;
        const double B = aRight / h;
        const double C = A + B + d * h;
        const double F = phi * h;

        const int idx = i - 1;
        diagonal[idx] = C;
        if (idx > 0) {
            lower[idx - 1] = -A;
        }
        if (idx < interior - 1) {
            upper[idx] = -B;
        }

        rhs[idx] = F;
        if (i == 1) {
            rhs[idx] += A * mu1;
        }
        if (i == interior) {
            rhs[idx] += B * mu2;
        }
    }

    std::vector<double> inner = solveTridiagonal(lower, diagonal, upper, rhs);
    std::vector<double> full(n + 1, 0.0);
    full[0] = mu1;
    full[n] = mu2;
    for (int i = 0; i < interior; ++i) {
        full[i + 1] = inner[i];
    }
    return full;
}

struct ErrorInfo {
    double maxError = 0.0;
    double x = 0.0;
    int index = 0;
};

ErrorInfo computeMaxError(const std::vector<double>& grid, const AnalyticSolution& analytic) {
    ErrorInfo info;
    const int n = static_cast<int>(grid.size()) - 1;
    for (int i = 0; i <= n; ++i) {
        const double x = static_cast<double>(i) / static_cast<double>(n);
        const double diff = analytic.value(x) - grid[static_cast<size_t>(i)];
        const double absDiff = std::abs(diff);
        if (absDiff > info.maxError) {
            info.maxError = absDiff;
            info.x = x;
            info.index = i;
        }
    }
    return info;
}

}  // namespace

TaskResult runFirstDirichletTestTask(const InputData& input, const VariantData& variant) {
    TaskResult task = makeTaskStub(
        "first-dirichlet-test",
        "Первая краевая тестовая задача",
        "1. Тестовая",
        "u(0)=mu1, u(1)=mu2",
        "Метод баланса, тестовая задача с аналитическим решением",
        "Курпяков Алексей",
        makeTestTaskColumns());

    int n = std::max(2, input.segments);

    const double xi = variant.xi;
    const double mu1 = variant.mu1;
    const double mu2 = variant.mu2;

    const AnalyticSolution analytic(xi, mu1, mu2);
    std::ostringstream note;

    if (n > input.maxSegments) {
        n = input.maxSegments;
        note << "Внимание: заданное n превышает maxSegments, используем n = " << n << ".\n";
    }

    std::vector<double> v = solveGrid(n, xi, mu1, mu2);
    ErrorInfo error = computeMaxError(v, analytic);

    note << "Уравнение: d/dx(k*du/dx) - q*u = -f*, коэффициенты кусочно-постоянные.\n"
         << "k1=" << kLeft << ", k2=" << kRight << ", q1=" << qLeft << ", q2=" << qRight
         << ", f1=" << fLeft << ", f2=" << fRight << ".\n"
         << "n = " << n << ", epsilon_1 = " << std::scientific << std::setprecision(6)
         << error.maxError << ".\n"
         << "Максимальная ошибка в узле i = " << error.index
         << ", x = " << std::fixed << std::setprecision(6) << error.x << ".\n";

    task.status = "done";
    task.note = note.str();

    for (int i = 0; i <= n; i += std::max(1, input.tableStride)) {
        const double x = static_cast<double>(i) / static_cast<double>(n);
        const double u = analytic.value(x);
        const double v_i = v[static_cast<size_t>(i)];
        task.rows.push_back(TableRow{i, x, u, v_i, 0.0, u - v_i});
    }

    if (task.rows.empty() || task.rows.back().index != n) {
        const double u = analytic.value(1.0);
        const double v_i = v.back();
        task.rows.push_back(TableRow{n, 1.0, u, v_i, 0.0, u - v_i});
    }

    return task;
}