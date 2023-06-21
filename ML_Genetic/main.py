import numpy as np
# 0.чтение
with open('input.txt', 'r') as f:
    n = int(f.readline())  # кол-во задач
    nlevel = list(map(int, f.readline().split()))  # сложность
    ntime = list(map(float, f.readline().split()))  # оценочное время
    m = int(f.readline())  # кол-во разраб
    mcoef = []  # коэффиценты
    for i in range(m):
        mcoef.append(list(map(float, f.readline().split())))


class GeneticAlgorithm:

    def __init__(self,
                 level: np.ndarray,
                 time: np.ndarray,
                 coef: np.ndarray,
                 size: int,
                 max_len_individual: int,
                 size_population: int,
                 size_selection: int,
                 p_mutation_ind: float,
                 p_mutation_gen: float,
                 iterations: int
                 ):
        self.level = level
        self.time = time
        self.coef = coef  # точки для посещения
        self.size = size  # размер поля
        self.max_len_individual = max_len_individual  # количество шагов который делает каждая особь
        self.size_population = size_population  # количество особей
        self.size_selection = size_selection  # выживаемые особи во время селекции
        self.p_mutation_ind = p_mutation_ind  # вероятность мутации детей
        self.p_mutation_gen = p_mutation_gen  # вероятность мутации генов
        self.iterations = iterations
        self.rng = np.random.default_rng()  # генератор случайных элементов
        self.population = self.rng.integers(1, len(self.coef) + 1, size=(self.size_population, self.max_len_individual)) #!!!

    def total_time(self, creature, dev_coef, task_level, task_time):
        # Вычисляется общее время выполнения проекта с учетом установленных задач и коэффициентов разработчика.
        # Возвращает:
        #     float, общее время для завершения проекта. Вещественный тип данных.
        total_time = np.zeros(len(dev_coef))
        for i, developer in enumerate(creature):
            total_time[developer - 1] += task_time[i] * dev_coef[developer - 1][task_level[i] - 1]
        return np.max(total_time)

    def fitness(self, i) -> np.ndarray:
        self.fitnes = []
        for creature in self.population: #!!!
            self.fitnes.append(self.total_time(creature, self.coef, self.level, self.time))
        print(f"min fitness(gen:{i}) = {np.min(self.fitnes)}")

        return self.fitnes

    def selection(self,i) -> None:
        self.selected = self.population[np.argsort(self.fitness(i))[:int(-self.p_mutation_gen*self.size_population)]] #!!!1

    def crossover(self) -> None:
        new_count = self.size_population - self.size_selection
        parent1 = self.rng.integers(0, self.size_selection, size=new_count)
        parent2 = (self.rng.integers(1, self.size_selection, size=new_count) + parent1) % self.size_selection

        point = self.rng.integers(1, self.max_len_individual - 1, size=new_count)
        self.childs = np.where(
            np.arange(self.max_len_individual)[None] <= point[..., None],
            self.selected[parent1],
            self.selected[parent2]
        )

    def mutation(self) -> None:
        mut_childs_mask = self.rng.choice(2, p=(1 - self.p_mutation_ind, self.p_mutation_ind),
                                          size=len(self.childs)) > 0
        mut_childs = self.rng.integers(1, len(self.coef) + 1, size=(mut_childs_mask.sum(), self.max_len_individual)) ###!!!исправлено
        gen_childs_mask = self.rng.random(size=mut_childs.shape) <= self.p_mutation_gen
        self.childs[mut_childs_mask] = np.where(gen_childs_mask, mut_childs, self.childs[mut_childs_mask])

    def step(self) -> None:
        for i in range(self.iterations):
            self.selection(i)
            self.crossover()
            self.mutation()
            self.population = np.concatenate([self.selected, self.childs], axis=0)  # !!!
        # Завершаем скрещивание
        # Выбираем лучшую особь
        self.fitne = []
        for p in self.population:
            self.fitne.append(-self.total_time(p, self.coef, self.level, self.time))  # ищем минимум
        best_idx = np.argmax(self.fitne)
        return self.population[best_idx]



ga = GeneticAlgorithm(nlevel, ntime, mcoef, n, 1000, 200, 20, 0.5, 0.15, 200)
out = ga.step()
with open('output.txt', 'w') as f:
    f.write(' '.join(map(str, out)))

