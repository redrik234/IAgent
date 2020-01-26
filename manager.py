class Manager:
    # === Данные сессии ===
    _user_id = 2242
    _case_id = 2  # кейсы лабиринта
    _tournament_id = 0  # id турнира
    _hash_id = 0  # id контрольной игры

    # === Данные по картам ===
    _map_nums = []  # номера карт
    _attempts_count = 5  # количество попыток на одн карту
    _total_game_count = 10  # общее количество игр

    # === Параметры обучения агента===
    _alpha = 0  # фактор обучения
    _gamma = 0  # фактор дисконтирования
    _delta = 0  # коэффициент уменьшения alpha
    _batch_size = 10  # размер пакета обучения
    _loss = 0  # ошибка выбора действия агента

    # === Конструктор менеджера игр ===
    # session_data = [user_id, case_id, tournament_id, hash_id]
    # map_data = [map_nums, attempts_count, total_game_count]
    # nnet_data = [alpha, gamma. delta, batch_size]
    def __init__(self, session_data, map_data, nnet_data):
        self._user_id = session_data[0]
        self._case_id = session_data[1]
        self._tournament_id = session_data[2]
        self._hash_id = session_data[3]

        self._map_nums = map_data[0]
        self._attempts_count = map_data[1]
        self._total_game_count = map_data[2]

        self._alpha = nnet_data[0]
        self._gamma = nnet_data[1]
        self._delta = nnet_data[2]
        self._batch_size = nnet_data[3]
