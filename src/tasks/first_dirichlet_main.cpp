#include "tasks/first_dirichlet_main.hpp"
#include "task_utils.hpp"
#include "solver.hpp"

#include <cmath>
#include <vector>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace {
    double k(double x, double xi) { return (x <= xi) ? 1.0 : std::exp(x * x); }
    double q(double x, double xi) { return (x <= xi) ? (x * x) : (1.0 + x * x * x * x); }
    double f(double x, double xi) { return (x <= xi) ? (x * x - 1.0) : 1.0; }

    template <typename Func>
    double integrate(Func func, double a, double b, double xi, int steps = 200) {
        auto do_int = [&](double start, double end) {
            if (start >= end) return 0.0;
            double h = (end - start) / steps;
            double sum = 0.0;
            for (int i = 0; i < steps; ++i) {
                sum += func(start + h * (i + 0.5)) * h;
            }
            return sum;
        };

        if (xi > a && xi < b) {
            return do_int(a, xi) + do_int(xi, b);
        }
        return do_int(a, b);
    }
    std::vector<double> solveGrid(int n, double xi, double mu1, double mu2) {
        double h = 1.0 / n;
        int N = n - 1;
        std::vector<double> lower(std::max(0, N - 1), 0.0);
        std::vector<double> diag(N, 0.0);
        std::vector<double> upper(std::max(0, N - 1), 0.0);
        std::vector<double> rhs(N, 0.0);

        auto inv_k = [xi](double x) { return 1.0 / k(x, xi); };
        auto bind_q = [xi](double x) { return q(x, xi); };
        auto bind_f = [xi](double x) { return f(x, xi); };

        for (int i = 1; i <= N; ++i) {
            double x_i = i * h;

            double a_i   = h / integrate(inv_k, x_i - h, x_i, xi);
            double a_ip1 = h / integrate(inv_k, x_i, x_i + h, xi);
            double d_i   = integrate(bind_q, x_i - h / 2.0, x_i + h / 2.0, xi) / h;
            double phi_i = integrate(bind_f, x_i - h / 2.0, x_i + h / 2.0, xi) / h;

            double A_i = a_i / h;
            double B_i = a_ip1 / h;
            double C_i = A_i + B_i + d_i * h;
            double F_i = phi_i * h;

            int idx = i - 1;
            diag[idx] = C_i;
            if (idx > 0) lower[idx - 1] = -A_i;
            if (idx < N - 1) upper[idx] = -B_i;

            rhs[idx] = F_i;
            
            if (i == 1) rhs[idx] += A_i * mu1;
            if (i == N) rhs[idx] += B_i * mu2;
        }

        std::vector<double> v_inner = solveTridiagonal(lower, diag, upper, rhs);
        std::vector<double> v_full(n + 1, 0.0);
        v_full[0] = mu1;
        v_full[n] = mu2;
        for (int i = 0; i < N; ++i) {
            v_full[i + 1] = v_inner[i];
        }
        return v_full;
    }
}

TaskResult runFirstDirichletMainTask(const InputData& input, const VariantData& variant) {
    TaskResult task = makeTaskStub(
        "first-dirichlet-main",
        "Первая краевая основная задача",
        "2. Основная",
        "u(0) = 2, u(1) = 1",
        "Метод баланса, сравнение сеток n и 2n",
        "Папулина Юлия",
        makeMainTaskColumns()
    );
    int n = std::max(2, input.segments);
    int mult = std::max(2, input.refinementMultiplier);
    
    double xi = variant.xi;
    double mu1 = variant.mu1;
    double mu2 = variant.mu2;

    std::vector<double> v;
    std::vector<double> v2;
    double max_diff = 0.0;
    double max_x = 0.0;
    int max_i = 0;

    std::ostringstream noteStr;

    if (n > input.maxSegments) {
        n = input.maxSegments;
        noteStr << "Внимание: заданное n превышает maxSegments, используем n = " << n << ".\n";
    }

    if (1LL * n * mult > input.maxSegments) {
        noteStr << "Невозможно построить уточненную сетку для сравнения: n * "
                << mult << " = " << (1LL * n * mult)
                << " больше maxSegments = " << input.maxSegments << ".\n"
                << "Увеличьте maxSegments или уменьшите начальное n.";
        task.note = noteStr.str();
        task.status = "warning";
        return task;
    }

    v = solveGrid(n, xi, mu1, mu2);
    v2 = solveGrid(n * mult, xi, mu1, mu2);

    for (int i = 0; i <= n; ++i) {
        double diff = std::abs(v[i] - v2[i * mult]);
        if (diff > max_diff) {
            max_diff = diff;
            max_x = i * (1.0 / n);
            max_i = i;
        }
    }

    noteStr << "Для решения задачи использована равномерная сетка с числом разбиений n = " << n << ";\n"
            << "точность ε2 = " 
            << std::scientific << std::setprecision(3) << max_diff << ";\n";
    
    task.status = "done";

    noteStr << "максимальная разность численных решений в общих узлах сетки наблюдается в точке x = " 
            << std::fixed << std::setprecision(5) << max_x << " в узле i = " << max_i << ".\n\n";

    task.note = noteStr.str();

    for (int i = 0; i <= n; i += input.tableStride) {
        TableRow row;
        row.index = i;
        row.x = i * (1.0 / n);
        row.u = 0.0;
        row.v = v[i];
        row.v2 = v2[i * mult];
        row.difference = v[i] - v2[i * mult];
        task.rows.push_back(row);
    }

    if (n % input.tableStride != 0) {
        TableRow row;
        row.index = n;
        row.x = 1.0;
        row.u = 0.0;
        row.v = v[n];
        row.v2 = v2[n * mult];
        row.difference = v[n] - v2[n * mult];
        task.rows.push_back(row);
    }

    return task;
}
