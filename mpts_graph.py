import tkinter as tk
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
import math


class Task():

    def __init__(self, id, pos, work):
        self.id = id
        self.pos = pos
        self.work = work
        self.effort = 0
        self.status = "Not Started"
        self.processors = 0

    def __repr__(self):
        class_name = type(self).__name__
        return f"{class_name}(ID={self.id},pos={self.pos},work_remain={self.get_work_remaining()})"

    def get_work(self):
        return self.work
    
    def get_id(self):
        return self.id

    def get_pos(self):
        return self.pos
    
    def get_work_remaining(self):
        return self.work - self.effort

    def get_status(self):
        return self.status

    def add_processor(self):
        self.processors += 1

    def remove_processor(self):
        self.processors -= 1

    def speedup_func(self, processors):
        # may eventually implement other types
        return processors

    def advance_step(self):
        # effort added = time * speedup_func(processors)
        if self.status != "Complete":
            self.effort += self.speedup_func(self.processors)

    def set_status(self, status):
        self.status = status

class Edge():

    def __init__(self, id, weight, dest):
        self.id = id # (x,y)
        self.weight = weight
        self.dest = dest

    def __repr__(self):
        class_name = type(self).__name__
        return f"{class_name}(ID={self.id},weight={self.weight},dest={self.dest})"

    def get_weight(self):
        return self.weight

    def get_id(self):
        return self.id

    def get_dest(self):
        return self.dest

class Agent():

    TRAVEL_SPEED = 20

    def __init__(self, id, tasks, edges, d, agents, init_assn):
        self.id = id
        self.T = tasks
        self.E = edges
        self.D = d
        self.A = agents
        self.traversal_time_remaining = None
        self.current_assn = init_assn
        self.set_assignment(init_assn) # initialize at node 0
        init_assn.add_processor()
        init_assn.set_status("In Progress")

    def __repr__(self):
        class_name = type(self).__name__
        return f"{class_name}(ID={self.id},curr_assn={self.current_assn},trav_time={self.traversal_time_remaining})"

    def set_assignment(self, assn):
        self.current_assn = assn
        # if task, no action needed
        # else is edge, so update time left
        if type(self.current_assn) == Edge:
            self.traversal_time_remaining = self.current_assn.get_weight()

    def get_assignment(self):
        return self.current_assn

    def get_traversal_time_remaining(self):
        return self.traversal_time_remaining


    def update_position(self, loc):

        new_pos = nx.get_node_attributes(self.D, 'pos')
        new_pos[self.id] = loc
        nx.set_node_attributes(self.D, new_pos, 'pos')

    # Pick closest incomplete node
    def greedy_choice(self):
        #determine lowest-cost edge & move onto it
        lowest_cost = 1000
        next_assn = None # if all tasks are complete, no new assignment will be made
        for t in self.T:
            if t.get_status() != "Complete":
                #edge from current task to t
                e = self.E.get((self.current_assn.get_id(), t.get_id()))
                cost = e.get_weight()
                if cost < lowest_cost:
                    lowest_cost = cost
                    next_assn = e
        return next_assn

    # Pick shortest edge not in use
    def dispersion_choice(self):
        #determine lowest-occupied edge & move onto it
        lowest_occ = len(self.A)
        next_assn = None # if all tasks are complete, no new assignment will be made
        for t in self.T:
            if t.get_status() != "Complete":
                #edge from current task to t
                e = self.E.get((self.current_assn.get_id(), t.get_id()))                
                # only least-occupied edges are candidates
                occ = 0
                for a in self.A:
                    # if an agent is already assigned to this edge
                    if a.get_assignment() == e:
                        occ += 1

                if occ < lowest_occ:
                    lowest_occ = occ
                    next_assn = e

        return next_assn


    def advance_step(self): 
        current_assn = self.current_assn
        # if working on task that has just completed, move to edge
        # else if working on edge that has just completed, begin work on task
        if type(current_assn) == Task:
            #if task is complete, move toward next-best task 
            # else remain at task (no change)
            if current_assn.get_status() == "Complete":
                current_assn.remove_processor()

                # SELECT NEXT EDGE
                #next_assn = self.greedy_choice()
                next_assn = self.dispersion_choice()
                
                # if no nodes remain to work on, then return to 0
                if next_assn == None:
                    e = self.E.get((current_assn.get_id(), 0))
                    next_assn = e
                
                self.set_assignment(next_assn) # start on edge toward next task
                self.edge_move()
        else:
            self.edge_move()
        
    
    def edge_move(self):
        current_assn = self.current_assn
        if current_assn == None: return

        #check if have arrived at target
        if self.get_traversal_time_remaining() <= self.TRAVEL_SPEED:
            # have arrived, so change assn to dest task
            dest = current_assn.get_dest() # dest is int id of task
            self.set_assignment(self.T[dest])
            self.current_assn.add_processor() # add processor to task

            # move agent node to new task node
            loc = self.T[dest].get_pos() # loc is (x,y)

        else:
            # not arrived yet, so continue traveling
            self.traversal_time_remaining -= self.TRAVEL_SPEED

            # move agent along traversal space between nodes
            old_loc_id = current_assn.get_id()[0]
            new_loc_id = current_assn.get_id()[1]
            old_coord = self.T[old_loc_id].get_pos()
            new_coord = self.T[new_loc_id].get_pos()

            dx_o = (new_coord[0] - old_coord[0])
            dy_o = (new_coord[1] - old_coord[1])
            
            theta = math.atan2(dy_o,dx_o)

            dx = 20*math.cos(theta)
            dy = 20*math.sin(theta)
            
            pos = nx.get_node_attributes(self.D, 'pos')[self.id]
            loc = (pos[0] + dx, pos[1] + dy)

        self.update_position(loc)

    

class MPTS_Graph_Solver():

    def __init__(self):
        self.G = nx.DiGraph() # actual graph
        self.D = nx.DiGraph() # agent positions
        self.T = [] #coverage tasks
        self.E = {} #edge tasks
        self.A = [] #agents
        self.current_step = 0
        self.max_steps = 1000  # The maximum number of loop iterations

    # Generate random nodes
    def generate_random_tasks(self, num_tasks):
        coords = []
        for i in range(num_tasks):
            # task position value
            x, y = random.randint(50, 450), random.randint(50, 450)
            # task area value
            work = random.randint(10,50)
            self.T.append(Task(i, (x,y), work))
            coords.append((x, y))
        return coords
    
    # Generate nodes from file
    def generate_tasks_from_file(self, filepath):
        cov_tasks = []

        return cov_tasks

    # Create nodes & generate edges
    def generate_graph(self, cities, tour=None):
        num_tasks = []
        for i, (x, y) in enumerate(cities):
            self.G.add_node(i, pos=(x, y))
            num_tasks.append(i)

        # generate edges
        for i in range(len(cities)):
            for j in num_tasks:
                if j == i: continue
                city1 = cities[i]
                city2 = cities[j]
                distance = round(((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)**0.5 + random.randint(10,30)) # random for flow asymmetry
                self.G.add_edge(i, j, weight=distance)
                self.E[(i,j)] = Edge((i,j),distance,j)

    # Calculate the total distance for a TSP tour
    def calculate_total_distance(self, tour, cities):
        total_distance = 0
        for i in range(len(tour) - 1):
            city1_idx = tour[i]
            city2_idx = tour[i + 1]
            city1 = cities[city1_idx]
            city2 = cities[city2_idx]
            distance = ((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)**0.5
            total_distance += distance
        return total_distance

    # Solve the TSP using brute force (permutations)
    def tsp_brute_force(self, cities):
        from itertools import permutations
        num_cities = len(cities)
        best_tour = None
        best_distance = float('inf')
        for tour in permutations(range(num_cities)):
            if tour[0] != 0: continue
            tour = tour + (0,)
            distance = self.calculate_total_distance(tour, cities)
            if distance < best_distance:
                best_distance = distance
                best_tour = tour

        return best_tour

    # Create a graph from a given number of cities
    def init_mpts(self, agents=5):
        self.current_step = 0
        self.T = []
        self.E = {}
        self.A = []

        #generate graph & tasks
        coords = self.generate_random_tasks(int(num_cities_entry.get())) #builds T
        self.generate_graph(coords) #builds E

        num_agents = int(num_agents_entry.get())

        #Display graph
        ax.clear()
        ax.set_title('Environment')
        pos = nx.get_node_attributes(self.G, 'pos')
        edge_labels = nx.get_edge_attributes(self.G, 'weight')
        sizes = []
        for t in self.T:
            sizes.append((t.get_work_remaining() ** 2) * 10) # scale node sizes according to work required

        nx.draw(self.G, pos, with_labels=True, ax=ax, node_size=sizes, node_color='lightblue', connectionstyle='arc3, rad = 0.1')
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, label_pos = 0.75, font_size = 6, alpha = 0.5, ax=ax)
        
        # Generate agents
        for i in range(num_agents):
            self.A.append(Agent(i,self.T,self.E,self.D,self.A,self.T[0])) # initialize at 0
            position = self.T[0].get_pos()
            self.D.add_node(i, pos=position)
        
        canvas.draw()

    def check_solved(self):
        solved = True
        for t in self.T:
            if t.get_status() != "Complete":
                solved = False
                break
        for a in self.A:
            if a.get_assignment() != None: # returned to 0?
                solved = False
                break
        return solved


    # Update the graph when the "Solve" button is clicked
    def solve_mpts(self):
        #print(self.T)
        #print(self.A)

        # Check if solved & agents all at 0        
        if self.check_solved(): return

        # Otherwise is unsolved, so continue with steps
        if self.current_step < self.max_steps:
            
            # Increment the step counter
            step_label.config(text=f'Step: {self.current_step}')
            self.current_step += 1

            # advance each agent (method for advancement will be optimized)
            for a in self.A:
                a.advance_step()
            
            # Prepare node categories
            # if not started, put in "not started" list
            not_started = []
            sizes_nostart = []
            # if in progress, put in "in progress" list
            in_progress = []
            sizes_prog = []
            # else if complete, put in "complete" list
            complete = []
            sizes_comp = []
            # for each task
            for t in self.T:
                # consider how many agents are assigned and what time is left
                t.advance_step()

                # Update task condition and prepare graphing
                size = (t.get_work() ** 2) * 10
                if t.get_work_remaining() <= 0:
                    t.set_status("Complete")
                    complete.append(t.get_id())
                    sizes_comp.append(size)
                elif t.get_work_remaining() < t.get_work():
                    t.set_status("In Progress")
                    in_progress.append(t.get_id())
                    sizes_prog.append(size)
                else:
                    t.set_status("Not Started")
                    not_started.append(t.get_id())
                    sizes_nostart.append(size)

            # Update Graph
            ax.clear()
            ax.set_title('Environment')
            pos = nx.get_node_attributes(self.G, 'pos') # city dictionary
            edge_labels = nx.get_edge_attributes(self.G, 'weight') # weight dictionary

            options = {"edgecolors": "tab:gray", "alpha": 0.9}
            # Add node types & labels
            nx.draw_networkx_nodes(self.G, pos, nodelist=complete, node_size=sizes_comp, node_color="tab:green", **options)
            nx.draw_networkx_nodes(self.G, pos, nodelist=in_progress, node_size=sizes_prog, node_color="tab:red", **options)
            nx.draw_networkx_nodes(self.G, pos, nodelist=not_started, node_size=sizes_nostart, node_color="tab:blue", **options)
            nx.draw_networkx_labels(self.G, pos)

            nx.draw_networkx_edges(self.G, pos, connectionstyle='arc3, rad = 0.1' )
            nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, label_pos = 0.65, font_size = 8)

            # Update agents on graph
            pos = nx.get_node_attributes(self.D, 'pos')
            nx.draw_networkx_nodes(self.D, pos, node_size=10, node_color="black")

            # TODO store current agent activities in ongoing schedule (display later)


            # Schedule the next iteration after a delay (e.g., 1000 milliseconds)
            canvas.draw()
            root.after(1000, self.solve_mpts)



mpts = MPTS_Graph_Solver()


# Create the main GUI window
root = tk.Tk()
root.title('MPTS')

# Create and place widgets in the GUI
frame = tk.Frame(root)
frame.pack(pady=10)

num_cities_label = tk.Label(frame, text='Number of Coverage Tasks:')
num_cities_label.pack(side=tk.LEFT)

num_cities_entry = tk.Entry(frame)
num_cities_entry.pack(side=tk.LEFT)

num_agents_label = tk.Label(frame, text='Number of Agents:')
num_agents_label.pack(side=tk.LEFT)

num_agents_entry = tk.Entry(frame)
num_agents_entry.pack(side=tk.LEFT)

init_button = tk.Button(frame, text='Init', command=mpts.init_mpts)
init_button.pack(side=tk.LEFT)

solve_button = tk.Button(frame, text='Solve', command=mpts.solve_mpts)
solve_button.pack(side=tk.LEFT)

#result_label = tk.Label(root, text='')
#result_label.pack()

step_label = tk.Label(root, text='Step: 0')
step_label.pack()


fig, ax = plt.subplots(figsize=(9, 9))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack()

root.mainloop()