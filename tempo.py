import pandas as pd
import numpy as np

# Пути к исходным файлам;
RUS_FILE = "rus.xlsx"      # русская речь монолингвов
BIL_FILE = "biling.xlsx"   # русская речь билингва
KAB_FILE = "kab.xlsx"      # кабардинская речь

def load_variant(path, variant_label):
    """
    Загрузка одного файла, нормализация названий столбцов,
    добавление метки языкового варианта.
    """
    df = pd.read_excel(path)

    # Приведение названий столбцов к единому виду: нижний регистр, подчеркивания вместо пробелов
    cols = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=cols)

    # Унификация возможных вариантов написания столбцов
    if "persent_duration" in df.columns and "percent_duration" not in df.columns:
        df = df.rename(columns={"persent_duration": "percent_duration"})
    if "total_duration" not in df.columns and "totaldur" in df.columns:
        df = df.rename(columns={"totaldur": "total_duration"})
    if "transcription_phonemes" not in df.columns and "transcription" in df.columns:
        df = df.rename(columns={"transcription": "transcription_phonemes"})
    if "file_name" not in df.columns and "filename" in df.columns:
        df = df.rename(columns={"filename": "file_name"})

    df["variant"] = variant_label
    return df

def detect_emotion(name: str) -> str:
    """
    Определение типа эмоции по имени файла (столбец file_name).
    Предполагается, что в имени содержатся подстроки 'гнев', 'радост',
    'нейтр' или их эквиваленты.
    """
    s = str(name).lower()
    if "гнев" in s:
        return "гнев"
    if "радост" in s:
        return "радость"
    if "нейтр" in s or "neutr" in s:
        return "нейтральная"
    return "другое"

def prepare_full_dataframe():
    """
    Объединение данных из трех файлов в один датафрейм,
    отбор гласных и добавление признака эмоции.
    """
    rus = load_variant(RUS_FILE, "русский вариант")
    bil = load_variant(BIL_FILE, "русская речь билингва")
    kab = load_variant(KAB_FILE, "кабардинский язык")

    df = pd.concat([rus, bil, kab], ignore_index=True)
    if "transcription_phonemes" in df.columns:
        df = df[df["transcription_phonemes"].str.lower() != "consonant"]
    else:
        raise ValueError("Не найден столбец transcription_phonemes")

    # Определение типа эмоции
    if "emotion" not in df.columns:
        if "file_name" not in df.columns:
            raise ValueError("Для определения эмоции нужен столбец emotion или file_name")
        df["emotion"] = df["file_name"].apply(detect_emotion)

    # Проверка наличия столбца с процентной длительностью
    if "percent_duration" not in df.columns:
        raise ValueError("Не найден столбец percent_duration / persent_duration")

    # При наличии общей длительности высказывания восстанавливаем абсолютную длительность
    if "total_duration" in df.columns:
        df["dur_s"] = df["percent_duration"] / 100.0 * df["total_duration"]
    elif "duration" in df.columns:
        # если общей длительности нет, а есть только Duration, используем ее напрямую
        df["dur_s"] = df["duration"]
    else:
        df["dur_s"] = np.nan  # абсолютная длительность недоступна

    return df

def assign_zones(df):
    """
    Присвоение позиционной зоны каждому гласному в пределах высказывания:
    первые три гласные – 'начало', последние три – 'конец',
    остальные – 'середина'.
    """
    if "file_name" not in df.columns:
        raise ValueError("Для позиционного анализа необходим столбец file_name")

    # Сортировка внутри каждого высказывания в исходном порядке
    df = df.sort_values(by=["variant", "emotion", "file_name"]).copy()

    # Порядковый номер гласного в высказывании
    df["v_index"] = df.groupby(["variant", "emotion", "file_name"]).cumcount() + 1

    # Число гласных в каждом высказывании
    counts = df.groupby(["variant", "emotion", "file_name"])["v_index"].transform("max")

    def zone_for_row(row_count, row_index):
        if row_index <= 3:
            return "начало"
        if row_index > row_count - 3:
            return "конец"
        return "середина"

    df["zone"] = np.vectorize(zone_for_row)(counts, df["v_index"])
    return df

def compute_overall_table(df):
    """
    Таблица общих показателей по каждому языковому варианту и эмоции:
    средний процент длительности и стандартное отклонение,
    при наличии – средняя абсолютная длительность и ее стандартное отклонение.
    """
    agg_dict = {
        "percent_duration": ["mean", "std"]
    }
    if "dur_s" in df.columns:
        agg_dict["dur_s"] = ["mean", "std"]

    overall = (
        df.groupby(["emotion", "variant"])
          .agg(agg_dict)
          .reset_index()
    )

    # Имена столбцов
    overall.columns = [
        "Эмоция",
        "Вариант",
        "Средний_%_длительности",
        "SD_%",
        "Средняя_длительность_с",
        "SD_длительности_с",
    ] if "dur_s" in df.columns else [
        "Эмоция",
        "Вариант",
        "Средний_%_длительности",
        "SD_%",
    ]

    return overall

def compute_zone_table(df):
    """
    Таблица показателей по позиционным зонам (начало, середина, конец)
    для каждого языкового варианта и эмоции.
    """
    df_z = assign_zones(df)

    agg_dict = {
        "percent_duration": ["mean", "std"]
    }
    if "dur_s" in df_z.columns:
        agg_dict["dur_s"] = ["mean", "std"]

    zones = (
        df_z.groupby(["emotion", "variant", "zone"])
            .agg(agg_dict)
            .reset_index()
    )

    if "dur_s" in df_z.columns:
        zones.columns = [
            "Эмоция",
            "Вариант",
            "Зона",
            "Средний_%_длительности",
            "SD_%",
            "Средняя_длительность_с",
            "SD_длительности_с",
        ]
    else:
        zones.columns = [
            "Эмоция",
            "Вариант",
            "Зона",
            "Средний_%_длительности",
            "SD_%",
        ]

    return zones

if __name__ == "__main__":
    df_all = prepare_full_dataframe()
    overall_table = compute_overall_table(df_all)
    print("Общая таблица по эмоциям и языкам:")
    print(overall_table)
    zone_table = compute_zone_table(df_all)
    print("\nТаблица по позиционным зонам:")
    print(zone_table)

    pivot_summary = (
        zone_table
        .pivot_table(
            index=["Эмоция", "Зона"],
            columns="Вариант",
            values="Средний_%_длительности"
        )
        .reset_index()
    )
    print("\nСводная таблица средней процентной длительности (для вставки в статью):")
    print(pivot_summary)
