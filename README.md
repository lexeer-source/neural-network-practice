# Практическое задание Основы нейронных сетей

Три самостоятельных модуля: MLP на NumPy, подготовка данных через CLI,
поточно-процессный конвейер аугментации. Запускать команды из корня проекта.

Python 3.10+; зависимости: `python -m pip install -r requirements.txt`.

1. `python -m scripts.generate_samples`
2. `python -m part1_mlp.app.main`
3. `python -m part2_prepare.app.cli prepare --path data/input/images --ext png --output data/output/images.npy`
4. `python -m part2_prepare.app.cli doctor`
5. `python -m part3_parallel.app.performance --path data/input/images`

Пути и параметры задаются аргументами и переменными окружения.
Пример переменных находится в `.env.example`; он не загружается
автоматически. TensorFlow/PyTorch и GPU для проекта не требуются.
Данные синтетические, воспроизводимо создаются локально и не содержат
персональных сведений. Выходные файлы и журналы исключены из Git.

Проверки: `python -m unittest discover -s tests -v`.
