<h1 align="center">Лабораторная работа №3</h1>

<h2 align="center">Простые системы классификации изображений на
основе сверточных нейронных сетей</h2>

<p align="center"><b>Теоритическая база</b><p>

TorchVision предлагает предварительно обученные веса для каждой предоставленной архитектуры с использованием PyTorch Hub. PyTorch Hub поддерживает публикацию предобученных моделей (код модели + веса) в репозиторий на GitHub. Для этого нужно добавить в репозиторий один .py файл. В этом файле хранится список библиотек, которые нужны для обучения модели. Примеры таких файлов доступны для опубликованных в Hub-е моделей: torchvision, huggingface-bert и gan-model-zoo. Пользователю доступен следующий функционал:
- получить актуальный список опубликованных моделей;
- загрузить модель;
- получить список доступных методов для загруженной модели.

Мы будем рассматривать три архитектуры (`AlexNet`, `ResNet`, `DenseNet`).

<b>1) AlexNet</b>

В 2012 году сверточная нейронная сеть с архитектурой `AlexNet` смогла обогнать своих предшественников. Её основное отличие состоит в том, что у неё больше фильтров на слое и вложенных сверточных слоев, а также:
- В качестве функции активации вместо привычного `Tanh` используется `ReLU`, что способствует увеличению скорости.
- Для борьбы с переобучением вместо регуляризации используется дропаут.

`AlexNet` состоит из 8 слоёв (5 свёрточных слоя и 3 полносвязных).

<figure>
  <p align="center"><img src="img_for_readme/alexnet.jpg"></p>
</figure>
<p align="center"><i>Архитектура AlexNet</i></p><br><br>

<b>2) ResNet</b>

В 2015 году `ResNet` заявила о себе, она произвела настоящую революцию глубины нейросетей и включала в себя 152 слоя! Ранее существовала серьёзная проблема с тем, что при увеличении слоёв у модели, она гораздо хуже поддаётся настройке и точность заметно начинает снижаться. Это связанно с тем, что слои в глубине модели очень сложно отслеживать и контролировать, а следовательно градиент начинает "исчезать". Создатели ResNet нашли способ решить эту проблему. Они решили не складывать слои друг на друга для изучения отображения нужной функции напрямую, а использовать остаточные блоки, которые пытаются «подогнать» это отображение. Таким образом `ResNet` стала первой <b>остаточной нейронной сетью</b>.
Другими словами `ResNet` "пропускает" некоторые слои. Они больше не содержат признаков и используются для нахождения остаточной функции H(x) = F(x) + x вместо того, чтобы искать H(x) напрямую.

<figure>
  <p align="center"><img src="img_for_readme/resnet.jpg"></p>
</figure>
<p align="center"><i>Архитектура ResNet</i></p><br><br>

<b>3) DenceNet</b>

`DenseNet` была предложена в 2017 году. Создатели вдохновлённые `ResNet`, которая показала всем, что существует возможность обучать глубокие нейронные сети путём "укороченного" соединения между слоями, придумали архитектуру `DenseNet`. Авторы проанализировав `ResNet` придумали <b>компактно соединенный блок/плотный блок (dense)</b>, который соединяет каждый слой с каждым другим слоем. Важно отметить, что, в отличие от `ResNet`, признаки прежде чем они будут переданы в следующий слой не суммируются, а конкатенируются в единый тензор. 
Подведя итог, можем сказать, что в `DenseNet` каждый слой получает дополнительные входные данные от всех предыдущих слоев и передает свои собственные признаки всем последующим слоям. При этом количество параметров сети `DenseNet` намного меньше, чем у сетей с такой же точностью работы. 

<figure>
  <p align="center"><img src="img_for_readme/densenet.jpg"></p>
</figure>
<p align="center"><i>Архитектура DenseNet</i></p><br><br>

<p align="center"><b>Описание разработанной системы</b><p>
Принцип работы состоит из пяти этапов:

1) Создание датасета.
2) Инициализация модели.
2) Инициализиция препроцессинга.
3) Использование препроцессинга для изображения.
4) Предсказывание класса предмета из изображения с помощью модели.
5) Вывод изображения с первыми 5-тью наиболее вероятными классами для предмета на изображении.

<b>Создание датасета</b>

Загружаем изображения с гугл диска и создаём список с предметом который присутствует на этих изображениях (всего 50 изображений).
```
images = []
for i in range(1,51):
  images.append(read_image(f"drive/MyDrive/dataset/img{i}.jpg").cuda())

text = ['acoustic guitar', 'microphone', 'moped', 'sports car', 'tennis ball', 'accordion', 'gondola', 'perfume', 'Polaroid camera', 'racket',
        'radio', 'quill', 'refrigerator', 'school bus', 'pillow', 'flute', 'oscilloscope', 'muzzle', 'maraca', 'kimono',
        'beacon', 'tiger shark', 'bakery', 'coho', 'barracouta', 'titi', 'bison', 'ice cream', 'trifle', 'pizza',
        'daisy', 'bolete', 'agaric', 'rapeseed', 'corn', 'dough', 'lemon', 'cucumber', 'broccoli','yawl',
        'tray', 'tractor', 'television', 'sax', 'pirate', 'pajama', 'obelisk', 'lifeboat', 'hook', 'goblet']
```
<figure>
  <p align="center"><img src="img_for_readme/dataset.jpg"></p>
</figure>
<p align="center"><i>Наш датасет</i></p><br><br>

<b>Инициализация модели</b>

В качестве примера я использую архитектуру `resnet50`.
Сначала импортируем `resnet50` и её веса `ResNet50_Weights` из библиотеки `torchvision.models`. Далее присваем модели веса обученные на датасете `IMAGENET1K_V1`. И вызываем метод `eval()`, означающий, что мы сделаем прямой проход по модели не изменяя её веса (другими словами без обучения).

```
from torchvision.models import resnet50, ResNet50_Weights
weights = ResNet50_Weights.IMAGENET1K_V1
model = resnet50(weights=weights).cuda()
model.eval()
```
<b>Инициализиция препроцессинга</b>

Инициализируем препроцессинг, для того чтобы все изображения,которые мы будем передавать в модель, имели необходимый для её работы формат (размер, цветовой режим и тд.)
```
preprocess = weights.transforms()
```

<b>Использование препроцессинга для изображения</b>

Применяем препроцессинг для каждого изображения.
```
batch = preprocess(images[i]).unsqueeze(0)
```
<b>Предсказывание класса предмета из изображения с помощью модели</b>

Делаем предсказание и записываем в переменную `prediction`.
Возвращаем в переменную `top_prediction` тензор из 5-ти наиболее вероятных классов для предмета на изображении. 
Создаём список `top_prediction_idx`, доставая из тензора индексы 5-ти наиболее вероятных классов. 
Создаём список `all_score`, в которых будет хранится вероятность для каждого из 5-ти наиболее вероятных классов. 
Создаём список `category_names`, в который запишем имена 5-ти наиболее вероятных классов. 
Далее создадим список `array_description`, содержащий описание для каждого изображения, и сохраним этот список в общий список `description` всего датасета.
```
prediction = model(batch).squeeze(0).softmax(0)
top_prediction = torch.topk(prediction.flatten(), N_top_predition).indices
top_prediction_idx = [e.item() for e in top_prediction]
all_score = [prediction[e].item() for e in top_prediction_idx]
category_names = [weights.meta["categories"][e] for e in top_prediction_idx]
array_description = [f"{top_prediction_idx[e]} {category_names[e]}: {100 * all_score[e]:.1f}%" for e in range(N_top_predition)]
description.append("\n".join(array_description))
```
<b> Вывод изображения с первыми 5-тью наиболее вероятными классами для предмета на изображении</b>

С помощью библиотеки matplotlib выведем поочерёдно все изображения с их описаниями. 
```
from matplotlib import pyplot as plt
import numpy as np
import torchvision.transforms as T

transform = T.Resize((500,500))

if type(images[0]) == torch.Tensor:
  images = [i.cpu() for i in images]
  images = [transform(i).data.numpy().transpose((1,2,0)) for i in images]
  
for i in range(len(images)):

  fig = plt.figure(i,figsize=(10.8, 4.8))
  fig.suptitle(f"{text[i]}", fontsize=25, y=0.95)

  # Левая картинка
  ax1 = fig.add_subplot(121)
  ax1.imshow(np.full((images[i].shape[0],images[i].shape[1],3), 255, dtype= 'uint8'))

  left, width = 0, images[i].shape[1]
  bottom, height = 0, images[i].shape[0]
  right = left + width
  top = bottom + height

  ax1.text(0.5 * (left + right), 0.5 * (bottom + top), 
        description[i],
        verticalalignment  = 'center', #  вертикальное выравнивание
        multialignment = 'left', #  текст начинается слева
        horizontalalignment = 'center',    #  горизонтальное выравнивание
        color = 'black',
        fontsize = 16)
  ax1.axis('off')

  # Правая картинка
  ax2 = fig.add_subplot(122)
  ax2.imshow(images[i])
  ax2.set_xticks([]),ax2.set_yticks([])

plt.show()
```
<figure>
  <p align="center"><img src="img_for_readme/output.jpg"></p>
</figure>
<p align="center"><i>Вывод результата (только первые два изображения)</i></p><br><br>


<p align="center"><b>Результаты работы и тестирования системы</b><p>
Протестируем 3 архитектуры сравнив их точность, скорость работы и потребление памяти.
Начнём тестирование с точности для этого мы напишем следующий код.
Для каждого изображения:

```
top1_accuracy += 1 if weights.meta["categories"][top_prediction_id[0]] == text[i] else 0
top5_accuracy += 1 if text[i] in [weights.meta["categories"][e] for e in top_prediction_id] else 0
```
Для вычисления итоговой точности модели:

```
top1_accuracy = top1_accuracy/len(images)
top5_accuracy = top5_accuracy/len(images)
print(f"top-1 accuracy: {100 * top1_accuracy:.1f}%")
print(f"top-5 accuracy: {100 * top5_accuracy:.1f}%")
```
Результат:

<figure>
  <p align="center"><img src="img_for_readme/acc.jpg"></p>
</figure>
<p align="center"><i>Метрики точности модели</i></p><br><br>

Далее протестируем среднее время выполнения подключив библиотеку `time` и создав с помощью неё список `test`.
```
import time
start = time.time()
prediction = model(batch).squeeze(0).softmax(0)
end = time.time()
test.append(end-start)
```
Результат:

<figure>
  <p align="center"><img src="img_for_readme/time.jpg"></p>
</figure>
<p align="center"><i>Среднее время выполнения</i></p><br><br>

Последний параметр, который мы будем использовать для сравнения архитектур – это потребление памяти. Нам поможет встроенные в `Pytorch` методы, которые называются `torch.cuda.max_memory_allocated()` и `torch.cuda.reset_peak_memory_stats()`, которые мы вставим в конец цикла. Первый метод записывает максимальное потребление памяти. Второй метод сбрасывает максимальное потребление памяти, это помогает нам считать потребление памяти в каждой итерации цикла отдельно. В итоге, получился следующий код: 
```
memory_test = torch.cuda.max_memory_allocated()
torch.cuda.reset_peak_memory_stats()
```
Результат:

<figure>
  <p align="center"><img src="img_for_readme/memory.jpg"></p>
</figure>
<p align="center"><i>Потребление памяти</i></p><br><br>

<h4 align="center">Выводы по работе</h4>

Вывод: Рассмотрев три архетектуры (`resnet50`, `alexnet`, `densenet161`), мы можем сделать вывод, что наиболее точные из них это `resnet50`, `densenet161`. Конечно это зависит от размера нашего датасета, скорее всего точность этих двух моделей будет немного другой. Хотя `alexnet` и уступает в точности остальным архитектурам, она имеет явное преимущество в скорости выполнения, а так же в количестве потребляемой памяти.


P.S. Для адекватного сравнения архитектур можем посмотреть на документацию PyTorch.
<figure>
  <p align="center"><img src="img_for_readme/doc.jpg"></p>
</figure>
<p align="center"><i>Документация PyTorch</i></p><br><br>
