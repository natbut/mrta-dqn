import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

APPROACHING = 0
WORKING = 1
IDLE = 2


class Environment:
    def __init__(
        self,
        num_agents,
        tasks_vs_reqd_agents,
        tasks_vs_reqd_time,
        task_to_task_transition_times=None,
        starting_task_id=0,
    ) -> None:
        """
        @param list tasks_vs_reqd_agents: how many agents are required to complete each task
        @param list tasks_vs_reqd_time: how many consecutive timesteps agents need to be on a task to complete it
        @param matrix task_to_task_transition_times: transition time between each task pair
        @param int starting_task_id: which task id agents start from (default = 0)
        """

        # set the environment constants
        self.env_con_num_agents = num_agents
        self.env_con_tasks_vs_reqd_agents = tasks_vs_reqd_agents
        self.env_con_tasks_vs_reqd_time = tasks_vs_reqd_time
        self.env_con_task_to_task_transition_times = task_to_task_transition_times
        self.env_con_starting_task_id = starting_task_id

        # declare the environment variables based on the constants
        self.env_var_agents_time_to_complete = [
            0 for agent in range(self.env_con_num_agents)
        ]
        self.env_var_agents_time_to_reach = [
            0 for agent in range(self.env_con_num_agents)
        ]
        self.env_var_agents_cum_idle_time = [
            0 for agent in range(self.env_con_num_agents)
        ]
        self.env_var_agents_prev_task = [
            starting_task_id for agent in range(self.env_con_num_agents)
        ]
        self.env_var_tasks_completed = [
            0 for task in range(len(self.env_con_tasks_vs_reqd_agents))
        ]
        self.env_var_tasks_time_left = tasks_vs_reqd_time.copy()

        # initialise the decision variables as numpy matrices
        self.dec_var_agents_at_tasks = np.zeros(
            (num_agents, len(self.env_con_tasks_vs_reqd_agents)), np.bool8
        )  # maps which task each agent is associated with
        self.dec_var_agents_current_action = np.zeros(
            (num_agents, 3), np.bool8
        )  # maps which action each agent is currently performing

        self.dec_var_agents_at_tasks[
            :, self.env_con_starting_task_id
        ] = 1  # initialise all agents to starting task id
        self.dec_var_agents_current_action[:, 2] = 1  # initalise all agents to idle

    def display_env(self):
        print("dec_var_agents_at_tasks: \n", self.dec_var_agents_at_tasks)
        print("dec_var_agents_current_action: \n", self.dec_var_agents_current_action)
        print("env_var_agents_time_to_complete: ", self.env_var_agents_time_to_complete)
        print("env_var_agents_time_to_reach: ", self.env_var_agents_time_to_reach)
        print("env_var_agents_cum_idle_time: ", self.env_var_agents_cum_idle_time)
        print("env_var_agents_prev_task: ", self.env_var_agents_prev_task)
        print("env_var_tasks_completed: ", self.env_var_tasks_completed)
        print("env_var_tasks_time_left: ", self.env_var_tasks_time_left)
        print("-----------------------------")

    def step(self, agents_vs_actions):
        """
        based on the actions specified, move all agents and update environment variables
        """
        reward = np.zeros(self.env_con_num_agents + 1, np.int8)  # initialise reward
        # make sure the actions are specified for each agent
        if len(agents_vs_actions) != self.env_con_num_agents:
            raise IndexError("Number of actions doesn't equal number of agents.")

        # loop through each specified action to allot the action in the decision variables to each agent
        for agent_action, i in zip(agents_vs_actions, range(len(agents_vs_actions))):
            print(i, agent_action)

            # if agent is in 'approach' with >0 time left, it cannot change actions
            if (
                self.dec_var_agents_current_action[i][APPROACHING] == 1
                and self.env_var_agents_time_to_reach[i] > 0
            ):
                self.env_var_agents_time_to_reach[i] -= 1
                continue
            # if agent is in 'work' with >0 time left, it cannot change actions
            elif (
                self.dec_var_agents_current_action[i][WORKING] == 1
                and self.env_var_agents_time_to_complete[i] > 0
            ):
                reward[i] = 1
                continue

            # Update the previous task of the agent
            # Initialise the time to reach and time to complete
            if agent_action[1] == WORKING:
                self.env_var_agents_prev_task[i] = agent_action[0]
                self.env_var_agents_time_to_reach[i] = 0
                self.env_var_agents_time_to_complete[i] = self.env_var_tasks_time_left[
                    agent_action[0]
                ]
                reward[i] = 1
            elif agent_action[1] == IDLE:
                self.env_var_agents_prev_task[i] = agent_action[0]
                self.env_var_agents_time_to_reach[i] = 0
                self.env_var_agents_time_to_complete[i] = 0
            else:
                self.env_var_agents_prev_task[i] = np.argmax(
                    self.dec_var_agents_at_tasks[i]
                )  # where the agent was in the previous timestep
                self.env_var_agents_time_to_reach[
                    i
                ] = self.env_con_task_to_task_transition_times[
                    self.env_var_agents_prev_task[i]
                ][
                    agent_action[0]
                ]  # initiliase as the transition time
                self.env_var_agents_time_to_complete[i] = 0

            # in all other cases,
            # 1. first clear the current location and action of the agent
            self.dec_var_agents_at_tasks[i, :] = False  # set ith row to all zeroes
            self.dec_var_agents_current_action[i, :] = False  # same here

            # 2. second, set the task and assignment as per the action
            self.dec_var_agents_at_tasks[i][
                agent_action[0]
            ] = True  # updating the task id
            self.dec_var_agents_current_action[i][
                agent_action[1]
            ] = True  # update the assignment

        # now based on the decision variables, update the other environment variables
        # 1. find the number of agents at each task that are working. bitwise AND the task column with working column
        for task_id in range(len(self.dec_var_agents_at_tasks[0])):
            arr_working_agents = (
                self.dec_var_agents_at_tasks[:, task_id]
                & self.dec_var_agents_current_action[:, WORKING]
            )
            # if task requirements are being met and task is incomplete, decrease the time left counter for the task and each involved agent
            if (
                np.sum(arr_working_agents) >= self.env_con_tasks_vs_reqd_agents[task_id]
                and self.env_var_tasks_time_left[task_id] > 0
            ):
                self.env_var_tasks_time_left[task_id] -= 1
                for agent_i in range(len(self.dec_var_agents_at_tasks[:, task_id])):
                    if (
                        arr_working_agents[agent_i] == True
                    ):  # if agent_1 is working at that task
                        # update the agent time left counter
                        self.env_var_agents_time_to_complete[agent_i] -= 1

            # 2. if task is complete, mark so in the task complete table
            if self.env_var_tasks_completed[task_id] == 0:
                if self.env_var_tasks_time_left[task_id] <= 0:
                    self.env_var_tasks_completed[task_id] = 1
                    reward[-1] = 10

        # 3. for each agent that is not working, incremement the cum idle time
        for agent_i in range(self.env_con_num_agents):
            if (self.dec_var_agents_current_action[:, WORKING])[agent_i] == False:
                self.env_var_agents_cum_idle_time[agent_i] += 1

        return reward


class TaskVisualization:
    def __init__(self, env):
        self.transition_times = env.env_con_task_to_task_transition_times
        self.num_nodes = len(self.transition_times)
        self.G = nx.DiGraph()
        attr_list = [
            {"x": np.random.randint(0, 100), "y": np.random.randint(0, 100)}
            for i in range(self.num_nodes)
        ]
        nodes_w_attr = list(zip(range(self.num_nodes), attr_list))
        self.G.add_nodes_from(nodes_w_attr)

        # initialize w euclidean distances if no transitions available
        if self.transition_times is None:
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    if i != j:
                        self.G.add_edge(
                            i,
                            j,
                            weight=np.hypot(
                                self.G.nodes[i]["x"] - self.G.nodes[j]["x"],
                                self.G.nodes[i]["y"] - self.G.nodes[j]["y"],
                            ),
                        )
        else:
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    if i != j:
                        self.G.add_edge(i, j, weight=self.transition_times[i][j])
        self.pos = nx.spring_layout(self.G)
        print("self pos", self.pos)

    def _display_agents(
        self, agent_vs_task_mat, dot_colour="black", action=APPROACHING
    ):
        # Draw agents on tasks with staggered positions
        stagger_offset = 0.015  # Adjust this value to control the stagger amount
        for agent_id in range(agent_vs_task_mat.shape[0]):
            for task_id in range(agent_vs_task_mat.shape[1]):
                if agent_vs_task_mat[agent_id, task_id]:
                    if action == WORKING or action == IDLE:
                        # Calculate the staggered position for the agent on the task
                        x_pos, y_pos = self.pos[task_id]
                    else:
                        # calculate the position based on distance of approach and nummber of steps taken
                        next_task_x_pos, next_task_y_pos = self.pos[task_id]
                        prev_task_x_pos, prev_task_y_pos = self.pos[
                            env.env_var_agents_prev_task[agent_id]
                        ]
                        x_line, y_line = (
                            next_task_x_pos - prev_task_x_pos,
                            next_task_y_pos - prev_task_y_pos,
                        )
                        t_steps = env.env_con_task_to_task_transition_times[task_id][
                            env.env_var_agents_prev_task[agent_id]
                        ]
                        x_delta, y_delta = x_line / t_steps, y_line / t_steps
                        t_steps_left = env.env_var_agents_time_to_reach[agent_id]
                        x_pos, y_pos = prev_task_x_pos + x_delta * (
                            t_steps - t_steps_left
                        ), prev_task_y_pos + y_delta * (t_steps - t_steps_left)

                    x_pos_staggered = x_pos + stagger_offset * agent_id
                    y_pos_staggered = y_pos + stagger_offset * agent_id
                    # Draw a ring
                    plt.scatter(
                        x_pos_staggered,
                        y_pos_staggered,
                        marker="o",
                        s=35,
                        edgecolor=dot_colour,  # Color of the ring border
                        facecolors="none",  # Transparent fill
                        linewidth=1.0,  # Adjust this value to control the ring border width
                        label=f"Agent {agent_id}",
                        zorder=2,
                    )

    def update(self, env):
        # bitwise AND to isolate agents_vs_tasks for each action type
        working_agents_vs_tasks = np.mat(
            [
                env.dec_var_agents_at_tasks[:, col_id]
                & env.dec_var_agents_current_action[:, WORKING]
                for col_id in range(env.dec_var_agents_at_tasks.shape[1])
            ]
        ).T
        approaching_agents_vs_tasks = np.mat(
            [
                env.dec_var_agents_at_tasks[:, col_id]
                & env.dec_var_agents_current_action[:, APPROACHING]
                for col_id in range(env.dec_var_agents_at_tasks.shape[1])
            ]
        ).T
        idle_agents_vs_tasks = np.mat(
            [
                env.dec_var_agents_at_tasks[:, col_id]
                & env.dec_var_agents_current_action[:, IDLE]
                for col_id in range(env.dec_var_agents_at_tasks.shape[1])
            ]
        ).T

        print("working_agents_vs_tasks: \n", working_agents_vs_tasks)
        print("approaching_agents_vs_tasks: \n", approaching_agents_vs_tasks)
        print("idle agents vs task: \n", idle_agents_vs_tasks)

        # Clear the previous plot
        plt.clf()

        # Draw nodes and edges
        nx.draw_networkx_nodes(
            self.G, self.pos, node_size=1000, node_color="skyblue", node_shape="o"
        )
        labels = {i: str(i) for i in range(self.num_nodes)}
        nx.draw_networkx_labels(self.G, self.pos, labels, font_size=7)
        nx.draw_networkx_edges(self.G, self.pos)
        edge_labels = nx.get_edge_attributes(self.G, "weight")
        nx.draw_networkx_edge_labels(
            self.G, self.pos, edge_labels=edge_labels, font_size=7
        )

        self._display_agents(
            working_agents_vs_tasks, dot_colour="green", action=WORKING
        )
        self._display_agents(
            approaching_agents_vs_tasks, dot_colour="blue", action=APPROACHING
        )
        self._display_agents(idle_agents_vs_tasks, dot_colour="red", action=IDLE)

        plt.show(block=False)
        plt.pause(5)
        plt.close()


if __name__ == "__main__":
    transition_matrix = [[0, 3, 2, 4], [3, 0, 5, 6], [2, 5, 0, 1], [4, 6, 1, 0]]

    env = Environment(5, [0, 3, 2, 4], [0, 5, 4, 1], transition_matrix, 0)

    task_visualization = TaskVisualization(env)
    print(env.dec_var_agents_at_tasks)
    print(env.dec_var_agents_current_action, "\n")

    # List of parameters for the step function
    actions_list = [
        [
            (1, APPROACHING),
            (2, APPROACHING),
            (2, APPROACHING),
            (3, APPROACHING),
            (0, IDLE),
        ],
        [
            (1, APPROACHING),
            (2, APPROACHING),
            (2, APPROACHING),
            (3, APPROACHING),
            (0, IDLE),
        ],
        [
            (1, APPROACHING),
            (2, APPROACHING),
            (2, APPROACHING),
            (3, APPROACHING),
            (0, IDLE),
        ],
        [(1, APPROACHING), (2, WORKING), (2, WORKING), (3, APPROACHING), (0, IDLE)],
        [(1, WORKING), (2, WORKING), (3, APPROACHING), (3, WORKING), (0, IDLE)],
        [(1, WORKING), (2, WORKING), (3, APPROACHING), (3, WORKING), (0, IDLE)],
        [(1, WORKING), (2, WORKING), (3, APPROACHING), (3, WORKING), (0, IDLE)],
    ]

    # Loop through the list and apply the step function, display_env, and update functions
    for actions in actions_list:
        reward = env.step(actions)
        print("Reward: ", reward)
        env.display_env()
        task_visualization.update(env)
