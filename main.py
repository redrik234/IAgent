import iagent
import manager
import time
import random

# гиперпараметры ==============================================================

# case_id   описание
# 1         L-1(Поиск клада. Известный Лабиринт. 2 обвала. 1 монстр.)
# 6         L-2(Поиск клада. Неизвестный Лабиринт. Без обвалов. Без монстра.)
# 7         L-3-1(Поиск клада. Неизвестный Лабиринт. 1 обвал. Без монстра)
# 4         L-4-1 (Поиск клада. Неизвестный Лабиринт. 2 обвала. Без монстра. )
# 2         L-5 (Поиск клада. Неизвестный Лабиринт. 2 обвала. 1 монстр.)
# 11        L-3-2 (Поиск клада. Неизвестный Лабиринт. без обвалов, 1 монстр)
# 12        L-4-2 (Поиск клада. Неизвестный Лабиринт. 1 обвал. 1 монстр.)

case_id = 32 # id кейса задачи
user_id = 2242 # id пользователя
tid = 0
hashid = 0
session_data = [user_id, case_id, tid, hashid]

data_path = "./data/" # путь до папки с данными
agent_name = "Darth_Maul" # название агента
#url = "https://mooped.net/local/its/game/agentaction/" # url сервера

# параметры нейросети
#inp_N = 3+ 4 + 4*8  # кол-во чисел, описывающих состояние игры
inp_N = 3 + 4 + 4 * 8 + 1
hidden_N = 128   # произвольно подбираемое число
out_N = 9       # кол-во возможных действий агента
nnet_filename = data_path + agent_name + '.nn'
nnet_parms = [nnet_filename, inp_N, hidden_N, out_N]

# создать или загрузить агента, которого будут тренировать
agent = iagent.IAgent(nnet_parms)

# параметры обучения нейросети
alpha = 0.0130 # фактор обучения
gamma = 0.9 # фактор дисконтирования
delta = 0.00001# коэф-т уменьшения alpha
batch_size = 20

map_numbers1 = list(range(1,251))
map_numbers  = random.sample(map_numbers1, 10)

attempts_per_map = 10 # количество попыток на каждую карту

# запуск обучения ===================================================
map_parms = [map_numbers, attempts_per_map]
learn_parms = [alpha, gamma, delta, batch_size]

# создать менеджера и запустить проверку / обучение
obs = manager.Manager(session_data, map_parms, learn_parms)
agent.set_user_id(obs.get_user_id())
agent.set_case_id(obs.get_case_id())
start = time.time()
obs.train(agent, 1)
end = time.time()
print("Время обучения составило (мин) ", round((end - start)/60))