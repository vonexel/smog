import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLayer(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        """
        KAN-слой, комбинирующий базовую линейную трансформацию и нелинейное смещение,
        вычисляемое через B-сплайны.

        Аргументы:
            in_features (int): размер входного вектора;
            out_features (int): размер выходного вектора;
            grid_size (int): количество интервалов сетки для сплайна;
            spline_order (int): порядок сплайна;
            scale_noise (float): масштаб шума для инициализации сплайн-коэффициентов;
            scale_base (float): масштаб инициализации базовых весов;
            scale_spline (float): масштаб инициализации весов сплайна;
            enable_standalone_scale_spline (bool): использовать ли отдельный масштаб для сплайновых весов;
            base_activation (nn.Module): функция активации для базовой ветки;
            grid_eps (float): коэффициент для адаптивного формирования сетки (не используется в forward, т.к. не прошел тесты);
            grid_range (list): диапазон значений для сетки.
        """
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Создаем фиксированную сетку для каждого входного канала.
        h = (grid_range[1] - grid_range[0]) / grid_size
        # Сетка имеет размер: grid_size + 2 * spline_order + 1 точка
        grid = torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float) * h + grid_range[0]
        grid = grid.expand(in_features, -1).contiguous()  # (in_features, grid_size + 2*spline_order + 1)
        self.register_buffer("grid", grid)

        # Базовые веса (как в обычном nn.Linear)
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        # Сплайновые веса; для каждого выхода и входа – набор коэффициентов (размер: grid_size + spline_order)
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps  # для реализации update_grid (to do на перспективу)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            # Инициализация шума для сплайновых весов
            noise = (torch.rand(self.grid_size + 1, self.in_features,
                                self.out_features) - 0.5) * self.scale_noise / self.grid_size
            # Переводим функцию смещения в набор коэффициентов для сплайна
            # Используем метод curve2coeff, который решает задачу наименьших квадратов
            x_points = self.grid.t()[self.spline_order: -self.spline_order]  # (grid_size+1, in_features)
            coeff = self.curve2coeff(x_points,
                                     noise)
            # возвращает тензор (out_features, in_features, grid_size + spline_order)
            scale = 1.0 if self.enable_standalone_scale_spline else self.scale_spline
            self.spline_weight.data.copy_(scale * coeff)
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x):
        """
        Вычисляет базис B-сплайнов для входного тензора x.
        Аргументы:
            x: тензор формы (N, in_features)
        Возвращает:
            Базисные функции B-сплайнов, тензор формы (N, in_features, grid_size + spline_order)
        """
        # x: (N, in_features)
        N = x.size(0)
        # Расширяем сетку для каждого примера: (N, in_features, grid_points)
        grid = self.grid.unsqueeze(0).expand(N, -1, -1)
        x_exp = x.unsqueeze(2)  # (N, in_features, 1)
        # Начальное приближение – индикаторная функция, определяющая, в каком интервале находится x
        bases = ((x_exp >= grid[:, :, :-1]) & (x_exp < grid[:, :, 1:])).to(x.dtype)
        # Рекурсивное вычисление B-сплайнов по определению Кокса – де Бора
        # https://math.stackexchange.com/questions/52157/can-cox-de-boor-recursion-formula-apply-to-b-splines-with-multiple-knots
        for k in range(1, self.spline_order + 1):
            num1 = x_exp - grid[:, :, :-(k + 1)]
            den1 = grid[:, :, k:-1] - grid[:, :, :-(k + 1)]
            term1 = num1 / (den1 + 1e-8) * bases[:, :, :-1]

            num2 = grid[:, :, k + 1:] - x_exp
            den2 = grid[:, :, k + 1:] - grid[:, :, 1:-k]
            term2 = num2 / (den2 + 1e-8) * bases[:, :, 1:]
            bases = term1 + term2
        # Результат имеет форму (N, in_features, grid_size + spline_order)
        return bases.contiguous()

    def curve2coeff(self, x, y):
        """
        Вычисляет коэффициенты сплайна, интерполирующего заданные точки.
        Аргументы:
            x: точки интерполяции, тензор формы (M, in_features)
            y: значения функции в этих точках, тензор формы (M, in_features, out_features)
        Возвращает:
            Коэффициенты, тензор формы (out_features, in_features, grid_size + spline_order)
        """
        M = x.size(0)
        coeffs = []
        for i in range(self.in_features):
            xi = x[:, i].unsqueeze(1)  # (M, 1)
            # Для каждого входного канала вычисляем B-сплайн базис
            grid_i = self.grid[i]  # (grid_size + 2*spline_order + 1)
            basis = ((xi >= grid_i[:-1]) & (xi < grid_i[1:])).to(x.dtype)
            for k in range(1, self.spline_order + 1):
                num1 = xi - grid_i[:-(k + 1)]
                den1 = grid_i[k:-1] - grid_i[:-(k + 1)]
                term1 = num1 / (den1 + 1e-8) * basis[:, :-1]

                num2 = grid_i[k + 1:] - xi
                den2 = grid_i[k + 1:] - grid_i[1:-k]
                term2 = num2 / (den2 + 1e-8) * basis[:, 1:]
                basis = term1 + term2
            # basis имеет форму (M, grid_size + spline_order)
            yi = y[:, i, :]  # (M, out_features)
            # Решаем задачу наименьших квадратов: A * coeff = yi
            # coeff имеет форму (grid_size + spline_order, out_features)
            A = basis  # (M, grid_size+spline_order)
            sol = torch.linalg.lstsq(A, yi).solution  # (grid_size+spline_order, out_features)
            coeffs.append(sol.unsqueeze(0))  # (1, grid_size+spline_order, out_features)
        coeffs = torch.cat(coeffs, dim=0)  # (in_features, grid_size+spline_order, out_features)
        # Меняем порядок осей, чтобы получить итоговую форму (out_features, in_features, grid_size+spline_order)
        coeffs = coeffs.permute(2, 0, 1).contiguous()
        return coeffs

    @property
    def scaled_spline_weight(self):
        if self.enable_standalone_scale_spline:
            return self.spline_weight * self.spline_scaler.unsqueeze(-1)
        else:
            return self.spline_weight

    def forward(self, x):
        """
        Прямой проход:
            - Вычисляется базовый выход через обычный линейный слой (после активации)
            - Вычисляется нелинейное смещение через B-сплайны и сплайновые веса
            - Результаты суммируются
        Аргументы:
            x: входной тензор, размер последней размерности должен быть in_features.
        Возвращает:
            тензор с последней размерностью out_features
        """
        original_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)  # (N, in_features)
        # Базовая ветка: активация + линейное преобразование
        activated = self.base_activation(x_flat)
        base_output = F.linear(activated, self.base_weight)  # (N, out_features)
        # Ветка сплайна: вычисляем базис B-сплайнов и применяем линейное преобразование через свёртку с весами сплайна
        bspline = self.b_splines(x_flat)
        # (N, in_features, grid_size + spline_order)
        bspline = bspline.view(x_flat.size(0), -1)
        # (N, in_features*(grid_size+spline_order))
        spline_weight = self.scaled_spline_weight.view(self.out_features, -1)
        # (out_features, in_features*(grid_size+spline_order))
        spline_output = F.linear(bspline, spline_weight)
        # (N, out_features)
        output = base_output + spline_output
        output = output.view(*original_shape[:-1], self.out_features)
        return output
