from sumo_rl import TrafficSignal

class CustomTrafficSignal(TrafficSignal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_green_phase = None
        self.is_red = False

    def _build_phases(self):
        self.red_time: int = 6

        phases = self.sumo.trafficlight.getAllProgramLogics(self.id)[0].phases
        #print(phases) -> (Phase(duration=39.0, state='GGGggrrrrGGGggrrrr', minDur=13.0, maxDur=50.0), Phase(duration=6.0, state='yyyyyrrrryyyyyrrrr', minDur=6.0, maxDur=6.0), Phase(duration=3.0, state='rrrrrrrrrrrrrrrrrr', minDur=3.0, maxDur=3.0), Phase(duration=39.0, state='rrrrrGGGgrrrrrGGGg', minDur=5.0, maxDur=50.0), Phase(duration=6.0, state='rrrrryyyyrrrrryyyy', minDur=6.0, maxDur=6.0), Phase(duration=3.0, state='rrrrrrrrrrrrrrrrrr', minDur=3.0, maxDur=3.0))
        if self.env.fixed_ts:
            self.num_green_phases = len(phases) // 2  # Number of green phases == number of phases (green+yellow) divided by 2
            return

        self.green_phases = []
        self.yellow_dict = {}
        self.red_dict = {}
        for phase in phases:
            state = phase.state
            if "y" not in state and (state.count("r") + state.count("s") != len(state)): #黄を含まない かつ 全てが赤or停止ではない
                if phase.duration >= 8:
                    self.green_phases.append(self.sumo.trafficlight.Phase(60, state)) #それを緑リストに格納
        self.num_green_phases = len(self.green_phases)
        self.all_phases = self.green_phases.copy()

        for i, p1 in enumerate(self.green_phases): #green_phase
            for j, p2 in enumerate(self.green_phases): #duration, state, minDur, maxDur
                if i == j:
                    continue
                yellow_state = ""
                red_state = ""
                for s in range(len(p1.state)):
                    # if (p1.state[s] == "G" or p1.state[s] == "g") and (p2.state[s] == "r" or p2.state[s] == "s"):
                    #     yellow_state += "y" #あるフェーズとその次のフェーズが青→赤であれば、その場所を"y"にしたstateを作成
                    #     red_state += "r"
                    # else:
                    #     yellow_state += p1.state[s] #そうでなければ、そのままの信号を維持
                    #     red_state += p1.state[s]
                    if p1.state[s] == "G" or p1.state[s] == "g":
                        yellow_state += "y"
                        red_state += "r"
                    else:
                        yellow_state += p1.state[s]
                        red_state += p1.state[s]
                self.yellow_dict[(i, j)] = len(self.all_phases) #key: 黄が使われる場所, value: all_phasesに黄が追加される場所
                self.all_phases.append(self.sumo.trafficlight.Phase(self.yellow_time, yellow_state))
                self.red_dict[(self.yellow_dict[(i, j)], j)] = len(self.all_phases) #key: 赤が使われる場所, value: all_phasesに赤が追加される場所
                self.all_phases.append(self.sumo.trafficlight.Phase(self.red_time, red_state))

        programs = self.sumo.trafficlight.getAllProgramLogics(self.id)
        logic = programs[0]
        logic.type = 0
        logic.phases = self.all_phases
        self.sumo.trafficlight.setProgramLogic(self.id, logic)
        self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[0].state)


    def update(self):
        self.time_since_last_phase_change += 1
        if self.is_red and self.time_since_last_phase_change == self.red_time + self.yellow_time:
            # print("run r => g")
            self.sumo.trafficlight.setRedYellowGreenState(
                self.id, self.all_phases[self.green_phase].state
            )
            self.is_red = False
        elif self.is_yellow and self.time_since_last_phase_change == self.yellow_time:
            # print("run y => r")
            self.sumo.trafficlight.setRedYellowGreenState(
                self.id, self.all_phases[self.red_dict[(self.yellow_dict[(self.pre_green_phase, self.green_phase)], self.green_phase)]].state
            )
            self.is_yellow = False
            self.is_red = True


    def set_next_phase(self, new_phase: int):
        """
        Sets what will be the next green phase and sets yellow phase if the next phase is different than the current.

        Args:
            new_phase (int): Number between [0 ... num_green_phases]
        """
        new_phase = int(new_phase)
        if self.green_phase == new_phase or self.time_since_last_phase_change < self.yellow_time + self.min_green + self.red_time: #現状維持
            if self.is_red:
                # print("run red stay")
                self.next_action_time = self.env.sim_step + self.delta_time
            else:
                # print("run green stay")
                self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
                self.next_action_time = self.env.sim_step + self.delta_time
        else:
            # print("run g => y")
            self.sumo.trafficlight.setRedYellowGreenState(
                self.id, self.all_phases[self.yellow_dict[(self.green_phase, new_phase)]].state
            )
            self.pre_green_phase = self.green_phase
            self.green_phase = new_phase
            self.next_action_time = self.env.sim_step + self.delta_time
            self.is_yellow = True
            self.time_since_last_phase_change = 0