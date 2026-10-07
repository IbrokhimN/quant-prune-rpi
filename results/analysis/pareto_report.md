# Парето-анализ

Фронтов три. Основной (рисунок fig3) строится по размеру и latency при статистически неразличимой точности (95% ДИ перекрываются), его и стоит цитировать в статье. Строгий и робастный берут точность третьим критерием. Строгий сравнивает точные числа, в робастном ничьёй считается разница accuracy меньше 0.2 п.п. или latency меньше max(3%, разброс между раундами). Строгий фронт по точности реагирует на шум в единицы объектов из 10 000.


## numpy blas1


**Основной фронт, размер x latency (batch=1)**. Точность, 95% ДИ перекрываются (max нижней границы 98.96, min верхней 99.03)

| вариант | на фронте | кто доминирует |
|---|---|---|
| float32/int8_dyn | нет | float32/int8_wo |
| float32/int8_wo | **да** | - |
| prune70/fp32 | **да** | - |
| prune60/fp32 | нет | prune70/fp32 |
| prune50/fp32 | нет | prune60/fp32, prune70/fp32 |
| prune40/fp32 | нет | prune50/fp32, prune60/fp32, prune70/fp32 |
| prune30/fp32 | нет | prune40/fp32, prune50/fp32, prune60/fp32 |
| prune20/fp32 | нет | prune30/fp32, prune40/fp32, prune50/fp32 |
| prune10/fp32 | нет | prune20/fp32, prune30/fp32, prune40/fp32 |
| float32/fp32 | нет | prune10/fp32, prune20/fp32, prune30/fp32 |
| float32_ft3/fp32 | нет | prune10/fp32, prune20/fp32, prune30/fp32 |
| unstruct10/fp32 | нет | prune10/fp32, prune20/fp32, prune30/fp32 |
| unstruct20/fp32 | нет | prune10/fp32, prune20/fp32, prune30/fp32 |
| unstruct30/fp32 | нет | prune10/fp32, prune20/fp32, prune30/fp32 |
| unstruct40/fp32 | нет | prune10/fp32, prune20/fp32, prune30/fp32 |
| unstruct50/fp32 | нет | prune10/fp32, prune20/fp32, prune30/fp32 |
| unstruct60/fp32 | нет | prune10/fp32, prune20/fp32, prune30/fp32 |
| unstruct70/fp32 | нет | prune10/fp32, prune20/fp32, prune30/fp32 |

**Основной фронт, размер x latency (batch=32)**. Точность, 95% ДИ перекрываются (max нижней границы 98.96, min верхней 99.03)

| вариант | на фронте | кто доминирует |
|---|---|---|
| float32/int8_dyn | нет | float32/int8_wo |
| float32/int8_wo | **да** | - |
| prune70/fp32 | **да** | - |
| prune60/fp32 | нет | prune70/fp32 |
| prune50/fp32 | нет | prune60/fp32, prune70/fp32 |
| prune40/fp32 | нет | prune50/fp32, prune60/fp32, prune70/fp32 |
| prune30/fp32 | нет | prune40/fp32, prune50/fp32, prune60/fp32 |
| prune20/fp32 | нет | float32/int8_wo, prune30/fp32, prune40/fp32 |
| prune10/fp32 | нет | float32/int8_wo, prune20/fp32, prune30/fp32 |
| float32/fp32 | нет | float32/int8_wo, prune10/fp32, prune20/fp32 |
| float32_ft3/fp32 | нет | float32/int8_wo, prune10/fp32, prune20/fp32 |
| unstruct10/fp32 | нет | float32/int8_wo, prune10/fp32, prune20/fp32 |
| unstruct20/fp32 | нет | float32/int8_wo, prune10/fp32, prune20/fp32 |
| unstruct30/fp32 | нет | float32/int8_wo, prune10/fp32, prune20/fp32 |
| unstruct40/fp32 | нет | float32/int8_wo, prune10/fp32, prune20/fp32 |
| unstruct50/fp32 | нет | float32/int8_wo, prune10/fp32, prune20/fp32 |
| unstruct60/fp32 | нет | float32/int8_wo, prune10/fp32, prune20/fp32 |
| unstruct70/fp32 | нет | float32/int8_wo, prune10/fp32, prune20/fp32 |

**размер vs accuracy**. Строгий фронт, int8-dyn (float32/int8_dyn); structured (prune70/fp32); unstructured (unstruct30/fp32)

Робастный фронт (допуск 0.2 п.п.), float32/int8_dyn, float32/int8_wo


**latency b=1 vs accuracy**. Строгий фронт, structured (prune70/fp32); unstructured (unstruct30/fp32, unstruct70/fp32)

Робастный фронт (допуск 0.2 п.п.), prune40/fp32, prune50/fp32, prune60/fp32, prune70/fp32


**latency b=32 vs accuracy**. Строгий фронт, structured (prune70/fp32); unstructured (unstruct30/fp32, unstruct60/fp32)

Робастный фронт (допуск 0.2 п.п.), prune50/fp32, prune60/fp32, prune70/fp32


## numpy blasauto


**Основной фронт, размер x latency (batch=1)**. Точность, 95% ДИ перекрываются (max нижней границы 98.96, min верхней 99.03)

| вариант | на фронте | кто доминирует |
|---|---|---|
| float32/int8_dyn | нет | float32/int8_wo |
| float32/int8_wo | **да** | - |
| prune70/fp32 | **да** | - |
| prune60/fp32 | нет | prune70/fp32 |
| prune50/fp32 | нет | prune60/fp32, prune70/fp32 |
| prune40/fp32 | нет | prune50/fp32, prune60/fp32, prune70/fp32 |
| prune30/fp32 | нет | prune40/fp32, prune50/fp32, prune60/fp32 |
| prune20/fp32 | нет | prune30/fp32, prune40/fp32, prune50/fp32 |
| prune10/fp32 | нет | prune20/fp32, prune30/fp32, prune40/fp32 |
| float32/fp32 | нет | prune10/fp32, prune20/fp32, prune30/fp32 |
| float32_ft3/fp32 | нет | prune10/fp32, prune20/fp32, prune30/fp32 |
| unstruct10/fp32 | нет | prune10/fp32, prune20/fp32, prune30/fp32 |
| unstruct20/fp32 | нет | prune10/fp32, prune20/fp32, prune30/fp32 |
| unstruct30/fp32 | нет | prune10/fp32, prune20/fp32, prune30/fp32 |
| unstruct40/fp32 | нет | prune10/fp32, prune20/fp32, prune30/fp32 |
| unstruct50/fp32 | нет | prune10/fp32, prune20/fp32, prune30/fp32 |
| unstruct60/fp32 | нет | prune10/fp32, prune20/fp32, prune30/fp32 |
| unstruct70/fp32 | нет | prune10/fp32, prune20/fp32, prune30/fp32 |

**Основной фронт, размер x latency (batch=32)**. Точность, 95% ДИ перекрываются (max нижней границы 98.96, min верхней 99.03)

| вариант | на фронте | кто доминирует |
|---|---|---|
| float32/int8_dyn | нет | float32/int8_wo |
| float32/int8_wo | **да** | - |
| prune70/fp32 | **да** | - |
| prune60/fp32 | нет | prune70/fp32 |
| prune50/fp32 | нет | prune60/fp32, prune70/fp32 |
| prune40/fp32 | нет | prune50/fp32, prune60/fp32, prune70/fp32 |
| prune30/fp32 | нет | prune40/fp32, prune50/fp32, prune60/fp32 |
| prune20/fp32 | нет | float32/int8_wo, prune30/fp32, prune40/fp32 |
| prune10/fp32 | нет | float32/int8_wo, prune20/fp32, prune30/fp32 |
| float32/fp32 | нет | float32/int8_wo, prune10/fp32, prune20/fp32 |
| float32_ft3/fp32 | нет | float32/int8_wo, prune10/fp32, prune20/fp32 |
| unstruct10/fp32 | нет | float32/int8_wo, prune10/fp32, prune20/fp32 |
| unstruct20/fp32 | нет | float32/int8_wo, prune10/fp32, prune20/fp32 |
| unstruct30/fp32 | нет | float32/int8_wo, prune10/fp32, prune20/fp32 |
| unstruct40/fp32 | нет | float32/int8_wo, prune10/fp32, prune20/fp32 |
| unstruct50/fp32 | нет | float32/int8_wo, prune10/fp32, prune20/fp32 |
| unstruct60/fp32 | нет | float32/int8_wo, prune10/fp32, prune20/fp32 |
| unstruct70/fp32 | нет | float32/int8_wo, prune10/fp32, prune20/fp32 |

**размер vs accuracy**. Строгий фронт, int8-dyn (float32/int8_dyn); structured (prune70/fp32); unstructured (unstruct30/fp32)

Робастный фронт (допуск 0.2 п.п.), float32/int8_dyn, float32/int8_wo


**latency b=1 vs accuracy**. Строгий фронт, structured (prune70/fp32); unstructured (unstruct30/fp32, unstruct60/fp32, unstruct70/fp32)

Робастный фронт (допуск 0.2 п.п.), prune40/fp32, prune50/fp32, prune60/fp32, prune70/fp32


**latency b=32 vs accuracy**. Строгий фронт, structured (prune70/fp32); unstructured (unstruct30/fp32, unstruct60/fp32)

Робастный фронт (допуск 0.2 п.п.), prune50/fp32, prune60/fp32, prune70/fp32

