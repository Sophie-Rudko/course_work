# Подробный пересказ статей для курсовой работы
## Тема: Веб-приложение для прогнозирования волатильности рынка с учётом инвесторских настроений

---

## Статья 1. Volatility forecasting using Deep Learning and sentiment analysis

**Авторы:** V. Ncume, T.L. van Zyl, A. Paskaramoorthy  
**Ссылка:** https://arxiv.org/abs/2210.12464

### Подробный пересказ

Исследование посвящено прогнозированию волатильности фондового рынка с помощью гибридной модели, объединяющей глубокое обучение и анализ тональности текста. Авторы подчёркивают, что большинство работ в этой области фокусируются на прогнозе доходности, тогда как прогноз волатильности при помощи нейросетей и sentiment-данных изучен гораздо меньше, несмотря на его важность для управления рисками и портфельной оптимизации.

Модель состоит из двух этапов. На первом этапе свёрточная нейронная сеть (CNN) анализирует заголовки глобальных новостей, собранных с Reddit — по 27 заголовков на каждый торговый день. Текст предварительно обрабатывают: токенизация, удаление стоп-слов, преобразование в векторные представления с помощью word2vec. CNN с 128 фильтрами и глобальным max-pooling выдаёт оценку тональности. На втором этапе LSTM-сеть принимает на вход эту тональность и прошлодневную волатильность (аппроксимированную квадратом логарифмической доходности) и выдаёт прогноз волатильности на следующий день. Архитектура LSTM: 30 нейронов в скрытом слое, dropout 0.2. Авторы также экспериментировали с «сдвинутым» sentiment — когда в модель подаётся не прошлодневная, а текущая тональность, чтобы проверить, насколько текущие настроения помогают объяснить сегодняшнюю волатильность.

Эксперименты проведены на данных S&P 500 и пяти индексах BRICS (Ibovespa, RTS-50, Nifty-50, SHCOMP, JSE Top 40) за период примерно с 2008 по 2016 год. Волатильность прогнозировали на тестовом периоде, оценивая качество по RMSE и сравнивая с бенчмарками: GARCH(1,1), SVR и LSTM без sentiment. Результаты оказались разными по рынкам. На S&P 500 лучшую RMSE показала SVR, а LSTM с sentiment — немного хуже. На Ibovespa (Бразилия) и SHCOMP (Китай) LSTM с sentiment стал лучшей моделью. На RTS-50 и JSE Top 40 лучше всего работал LSTM с «сдвинутым» sentiment. Это указывает на то, что связь между настроениями и волатильностью зависит от конкретного рынка: в одних странах полезнее прошлые настроения, в других — текущие. CNN для sentiment-классификации превзошла логистическую регрессию и случайный лес по F-score (0,86 против 0,84–0,85), что подтверждает целесообразность использования нейросетей для анализа тональности новостей.

Авторы также отмечают ограничение: модель ориентирована на рыночные индексы, а не на отдельные активы. Для отдельных акций текст для анализа должен быть более специфичным (например, новости по конкретной компании), поскольку на них сильнее влияют идиосинкратические риски.

---

### Цитаты и пояснения к ним

**Цитата 1:**
> *Several studies have shown that deep learning models can provide more accurate volatility forecasts than the traditional methods used within this domain. This paper presents a composite model that merges a deep learning approach with sentiment analysis for predicting market volatility.*

**Почему выбрана:** Формулирует мотивацию и основной вклад работы — использование DL и sentiment для прогноза волатильности. Подходит для обоснования выбора подхода в курсовой.

---

**Цитата 2:**
> *Our results demonstrate that including sentiment can improve Deep Learning volatility forecasting models. However, in contrast to return forecasting, the performance benefits of including sentiment for volatility forecasting appears to be market specific.*

**Почему выбрана:** Кратко суммирует главный вывод: sentiment помогает, но эффект рыночно-специфичен. Полезно для обсуждения ограничений и условий применения модели.

---

**Цитата 3:**
> *Our volatility forecasting results demonstrated that sentiment input can add predictive power to a volatility forecasting model, but this appears to be market specific. Although the LSTM with sentiment did not outperform the benchmarks in some markets, it did provide more accurate forecasts than the LSTM without sentiment input.*

**Почему выбрана:** Уточняет: даже когда LSTM с sentiment не выигрывает у других методов, он всё равно превосходит LSTM без sentiment. Обосновывает полезность включения sentiment в модель.

---

**Цитата 4:**
> *We propose a model that merges the sentiment extracted from Reddit global news headlines with the previous time step's volatility to predict the volatility of market indices. Our hybrid model consists of two neural networks: a CNN to perform the sentiment analysis, and an LSTM to forecast volatility.*

**Почему выбрана:** Точно описывает архитектуру модели (CNN + LSTM) и источник данных. Можно использовать при описании методики в курсовой.

---

**Цитата 5:**
> *Lastly, the Convolutional Neural Network showed better results for sentiment classification - outperforming the benchmark classifiers of Logistic Regression and Random Forest (Table 8) with an F-score of 0.86.*

**Почему выбрана:** Даёт количественную оценку качества sentiment-модели (F-score 0,86). Подтверждает, что нейросеть подходит для классификации тональности.

---

**Цитата 6:**
> *Furthermore, the findings of our research do not imply that our model will perform in the same manner on individual financial assets. Individual financial assets have additional sources of variation due to idiosyncratic risks. Thus, textual data for sentiment analysis would have to be more specific than global news headlines.*

**Почему выбрана:** Обозначает границы применимости: для отдельных акций нужны более специализированные текстовые данные. Важно для постановки задачи и обсуждения ограничений в курсовой.

---

## Статья 2. Predicting Stock Prices with FinBERT-LSTM: Integrating News Sentiment Analysis

**Авторы:** Wenjun Gu, Yihao Zhong, Shizun Li и др.  
**Ссылка:** https://arxiv.org/abs/2407.16150

### Подробный пересказ

Статья посвящена прогнозированию цен акций с учётом новостей и их тональности. Авторы напоминают, что состояние рынка отражает состояние экономики, и потому предсказание рыночных трендов и анализ факторов, влияющих на цены, давно интересуют исследователей. Они используют сочетание исторических цен и статей финансовых, деловых и технических новостей, которые несут информацию о рынке.

Ключевой элемент — модель FinBERT, предобученная NLP-модель для анализа тональности финансовых текстов. FinBERT основана на BERT и дообучена на финансовых данных, поэтому лучше учитывает специфику терминологии и контекста в новостях и отчётах. Вместо простого применения FinBERT авторы интегрируют её с LSTM: тональность из FinBERT и временные ряды цен подаются в LSTM, которая выдаёт прогноз. Такая связка позволяет одновременно использовать и текстовую информацию, и историческую динамику цен.

Новости делятся на три категории по уровню охвата: рыночные (общий рынок), отраслевые и по отдельным акциям. Каждой категории присваивается вес, что позволяет модели по-разному учитывать влияние разных типов новостей. В качестве источника новостей используется Benzinga — агрегатор финансовых новостей, что по сути аналогично задаче агрегации через NewsAPI или RSS.

Эксперименты проведены на акциях индекса NASDAQ-100. Качество оценивалось по MAE, MAPE и Accuracy. Модель FinBERT-LSTM сравнивали с чистым LSTM и с обычной DNN. Результаты показывают устойчивое преимущество FinBERT-LSTM: она даёт лучшие прогнозы, за ней идёт LSTM, а DNN — на третьем месте. Это свидетельствует о том, что сочетание предобученной модели для текста (FinBERT) и рекуррентной сети для временных рядов (LSTM) повышает точность прогноза по сравнению с использованием только цен или более простых моделей.

---

### Цитаты и пояснения к ним

**Цитата 1:**
> *The stock market's ascent typically mirrors the flourishing state of the economy, whereas its decline is often an indicator of an economic downturn. Therefore, for a long time, significant correlation elements for predicting trends in financial stock markets have been widely discussed, and people are becoming increasingly interested in the task of financial text mining.*

**Почему выбрана:** Обосновывает актуальность анализа текста для прогноза рынка. Подходит для введения в курсовой: мотивация темы и важность финансового text mining.

---

**Цитата 2:**
> *In this article, we use deep learning networks, based on the history of stock prices and articles of financial, business, technical news that introduce market information to predict stock prices. We illustrate the enhancement of predictive precision by integrating weighted news categories into the forecasting model.*

**Почему выбрана:** Описывает общую идею: использование цен и новостей, а также взвешенных категорий новостей. Связана с задачей агрегации новостей и их структурирования по категориям.

---

**Цитата 3:**
> *We developed a pre-trained NLP model known as FinBERT, designed to discern the sentiments within financial texts. Subsequently, we advanced this model by incorporating the sophisticated Long Short Term Memory (LSTM) architecture, thus constructing the innovative FinBERT-LSTM model.*

**Почему выбрана:** Чётко определяет FinBERT и FinBERT-LSTM. Обосновывает выбор BERT/RoBERTa-подобной модели (FinBERT) и LSTM в твоей задаче.

---

**Цитата 4:**
> *This model utilizes news categories related to the stock market structure hierarchy, namely market, industry, and stock related news categories, combined with the stock market's stock price situation in the previous week for prediction.*

**Почему выбрана:** Описывает структуру входных данных: категории новостей (рынок, отрасль, акция) и прошлые цены. Полезно для визуализации и интерпретации связи sentiment и трендов в дашборде.

---

**Цитата 5:**
> *We selected NASDAQ-100 index stock data and trained the model on Benzinga news articles, and utilized Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and Accuracy as the key metrics for the assessment and comparative analysis of the model's performance. The results indicate that FinBERT-LSTM performs the best, followed by LSTM, and DNN model ranks third in terms of effectiveness.*

**Почему выбрана:** Содержит описание данных (NASDAQ-100, Benzinga), метрик (MAE, MAPE, Accuracy) и главный количественный вывод: FinBERT-LSTM превосходит LSTM и DNN. Подходит для раздела результатов и выводов.

---

## Связь статей с задачами курсовой

| Задача курсовой | Статья 1 (2210.12464) | Статья 2 (2407.16150) |
|-----------------|------------------------|------------------------|
| Агрегация новостей и анализ тональности | Reddit headlines, CNN | Benzinga, FinBERT |
| Прогноз волатильности S&P 500 | LSTM + sentiment, S&P 500 | — (NASDAQ-100) |
| LSTM / Attention | LSTM | FinBERT + LSTM |
| Корреляция sentiment и рыночных трендов | Волатильность vs sentiment | Взвешенные категории новостей |
