import random
import os

class Manager:
    # === Данные сессии ===
    __user_id = None  # id пользователя
    __case_id = None  # кейсы лабиринта
    __tournament_id = 0  # id турнира
    __hash_id = 0  # id контрольной игры

    # === Данные по картам ===
    __map_nums = []  # номера карт
    __attempts_count = 5  # количество попыток на одн карту

    # === Параметры обучения агента===
    __alpha = 0  # фактор обучения
    __gamma = 0  # фактор дисконтирования
    __delta = 0  # коэффициент уменьшения alpha
    __batch_size = 10  # размер пакета обучения
    __loss = 0  # ошибка выбора действия агента

    # === Параметры для подсчета результатов игр ===
    __games_count = 0
    __wins_count = 0
    __score_count = 0
    __lost_map_list = []
    __total_games_count = 0  # общее количество игр
    __gamesQ = 0

    # === Конструктор менеджера игр ===
    # session_data = [user_id, case_id, tournament_id, hash_id]
    # map_data = [map_nums, attempts_count, total_game_count]
    # nnet_data = [alpha, gamma. delta, batch_size]
    def __init__(self, session_data, map_data, nnet_data):
        self.__user_id = session_data[0]
        self.__case_id = session_data[1]
        self.__tournament_id = session_data[2]
        self.__hash_id = session_data[3]

        self.__map_nums = map_data[0]
        self.__attempts_count = map_data[1]
        # self.__total_games_count = map_data[2]

        self.__alpha = nnet_data[0]
        self.__gamma = nnet_data[1]
        self.__delta = nnet_data[2]
        self.__batch_size = nnet_data[3]

    def get_user_id(self):
        return self.__user_id

    def get_case_id(self):
        return self.__case_id

    # === Метод для обучения интеллектуального агента ===
    def train(self, agent, iter_count=1):
        self.__print_beginning_of_games(iter_count)

        self.__lost_map_list = []

        for it in range(1, iter_count + 1):
            self.__games_count = 0
            self.__wins_count = 0
            self.__score_count = 0

            agent = self.__play(agent, it)

            self.__total_games_count += self.__games_count
            win_rate = (self.__wins_count * 100) / self.__games_count
            score_rate = self.__score_count / self.__games_count
            self.__print_overall_result(win_rate, score_rate, agent.get_help_degree())

    def __play(self, agent, iteration):
        random.shuffle(self.__map_nums)
        for map_num in range(len(self.__map_nums)):
            for attempt_num in range(1, self.__attempts_count + 1):
                hash_id, map_id = self.__get_hash_and_map(map_num)
                code, score, self.__gamesQ = agent.playGame(map_id, self.__gamesQ, self.__tournament_id, self.__hash_id)
                if code is None:
                    print("Connection failed for: map = {0}, attempt = {1}".format(map_id, attempt_num))
                else:
                    self.__update_games_result(code, score, map_id)

                self.__print_game_result(map_num, attempt_num, iteration)

                self.__alpha -= self.__delta
                if self.__alpha < 0.01:
                    self.__alpha = 0.01
                agent.__saveNnet__()
            #agent.update_nnet(self.__alpha, self.__gamma, self.__batch_size)

        return agent

    def __print_overall_result(self, win_rate, score_rate, help_degree):
        games_info = "* Games: " + str(self.__games_count)
        win_info = "* Win rate: {:6.2f}, Score rate: {:6.2f}" . format(win_rate, score_rate)
        lost_info = "* Lost Game Maps: " + ', '.join(str(e) for e in self.__lost_map_list)
        if len(win_info) > len(lost_info):
            str_length = len(win_info) + 1
        else:
            str_length = len(lost_info) + 1

        print('*' * (str_length + 1))
        s = ' ' * (str_length - len(games_info)) + '*'
        print(games_info + s)
        s = ' ' * (str_length - len(win_info)) + '*'
        print(win_info + s)
        s = ' ' * (str_length - len(lost_info)) + '*'
        print(lost_info + s)
        print('*' * (str_length + 1))

        f = open('log.txt', 'a')
        f.write('*' * (str_length + 1) + '\n')
        f.write(f"* Games: {self.__games_count}, CaseId: {self.__case_id} \n")
        f.write(f"* Win rate: {win_rate}, Score rate: {score_rate} \n")
        f.write("* Lost Game Maps: " + ', '.join(str(e) for e in self.__lost_map_list))
        f.write(f"help_degree:{help_degree}")
        f.write('\n' + '*' * (str_length + 1) + '\n')
        f.close()

    def __print_game_result(self, map_num, attempt_num, iteration):
        print('*' * 80)
        if self.__tournament_id == 0:
            print(
                "map_num: ", map_num + 1,
                ", attempt: ", attempt_num,
                ", wins=", self.__wins_count,
                ", alpha = {:8.6f}".format(self.__alpha)
            )
        else:
            print(
                "Iteration: ", iteration,
                ", tid: ", self.__tournament_id,
                ", wins=", self.__wins_count,
                ", alpha = {:8.6f}".format(self.__alpha)
            )

    def __print_beginning_of_games(self, iter_count):
        print('=' * 80)
        print(' ' * 15 + "!!!Start of torture for the agent!!!")
        print(
            "Iteration count: ", iter_count,
            ", attempt count: ", self.__attempts_count,
            ", total game count: ", self.__total_games_count
        )
        print('=' * 80)

    def __get_hash_and_map(self, map_num):
        hash_id, map_id = None, None
        if self.__hash_id == 0:
            hash_id = 0
            map_id = self.__map_nums[map_num]
        else:
            hash_id = self.__hash_id[map_num]
            map_id = 0
        return hash_id, map_id

    def __update_games_result(self, code, score, map_num):
        if code == 2:
            self.__wins_count += 1
            self.__score_count += score
        else:
            self.__lost_map_list.append(map_num)

        self.__games_count += 1
